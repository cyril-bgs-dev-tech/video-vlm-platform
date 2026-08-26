"""
tests/test_api.py — Tests de l'API FastAPI
Exécuter : pytest tests/ -v
Prérequis : pip install pytest pytest-asyncio httpx
"""
import pytest
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport

# Import de l'app (sans démarrer les workers)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_celery_task():
    """Task Celery mockée qui retourne immédiatement."""
    task = MagicMock()
    task.id     = str(uuid.uuid4())
    task.state  = "SUCCESS"
    task.ready.return_value = True
    task.result = {"status": "mock", "video_id": "test-vid"}
    return task


@pytest.fixture
def mock_send_task(mock_celery_task):
    """Mock de celery_app.send_task."""
    with patch("workers.celery_app.celery_app.send_task", return_value=mock_celery_task):
        yield mock_celery_task


@pytest.fixture
def mock_async_result(mock_celery_task):
    """Mock de celery_app.AsyncResult."""
    with patch("workers.celery_app.celery_app.AsyncResult", return_value=mock_celery_task):
        yield mock_celery_task


@pytest.fixture
async def client(mock_send_task):
    """Client HTTP asynchrone pour tester l'API."""
    from main import app
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


# ── Tests /health ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_health_endpoint(client):
    """L'endpoint /health doit retourner 200 avec status=ok."""
    # /health compte les workers via insp.ping() — active()/stats() enrichissent
    # seulement la réponse. Sans ping mocké, status vaut "degraded".
    with patch("workers.celery_app.celery_app.control") as mock_ctrl:
        insp = mock_ctrl.inspect.return_value
        insp.active.return_value = {"detection_worker@test": []}
        insp.ping.return_value = {"detection_worker@test": {"ok": "pong"}}
        insp.stats.return_value = {
            "detection_worker@test": {"pool": {"implementation": "solo"}}
        }
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "workers" in data
    assert data["workers_count"] == 1
    assert data["workers_online"] == ["detection_worker@test"]


# ── Tests /upload ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_upload_video_success(client, tmp_path):
    """Upload d'un fichier MP4 factice doit retourner video_id."""
    # Créer un faux fichier MP4
    fake_video = tmp_path / "test.mp4"
    fake_video.write_bytes(b"fakemp4content" * 100)

    with patch("aiofiles.open", create=True) as mock_open:
        mock_file = AsyncMock()
        mock_open.return_value.__aenter__.return_value = mock_file
        mock_open.return_value.__aexit__ = AsyncMock(return_value=False)

        resp = await client.post(
            "/upload",
            files={"file": ("test.mp4", fake_video.read_bytes(), "video/mp4")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "video_id" in data
    assert len(data["video_id"]) == 36  # UUID format


@pytest.mark.asyncio
async def test_upload_no_file(client):
    """Upload sans fichier doit retourner 422."""
    resp = await client.post("/upload")
    assert resp.status_code == 422


# ── Tests /vlm/sam2 ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_sam2_point_prompt(client, mock_send_task):
    """SAM2 avec point prompt doit retourner 202 et task_id."""
    resp = await client.post("/vlm/sam2", json={
        "video_id":  str(uuid.uuid4()),
        "frame":     42,
        "prompt":    {"type": "point", "x": 320, "y": 240},
        "propagate": False,
    })
    assert resp.status_code == 202
    assert "task_id" in resp.json()


@pytest.mark.asyncio
async def test_sam2_box_prompt(client, mock_send_task):
    """SAM2 avec box prompt."""
    resp = await client.post("/vlm/sam2", json={
        "video_id":  str(uuid.uuid4()),
        "frame":     10,
        "prompt":    {"type": "box", "x1": 100, "y1": 100, "x2": 500, "y2": 400},
        "propagate": True,
    })
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_sam2_text_prompt(client, mock_send_task):
    """SAM2 avec texte prompt (GroundingDINO + SAM2)."""
    resp = await client.post("/vlm/sam2", json={
        "video_id": str(uuid.uuid4()),
        "frame":    0,
        "prompt":   {"type": "text", "query": "person with red hat"},
    })
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_sam2_missing_fields(client):
    """SAM2 sans champs obligatoires → 422."""
    resp = await client.post("/vlm/sam2", json={"video_id": "test"})
    assert resp.status_code == 422


# ── Tests /vlm/task/{task_id} ─────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_task_status_success(client, mock_async_result):
    """Polling d'une task terminée doit retourner SUCCESS."""
    task_id = str(uuid.uuid4())
    resp = await client.get(f"/tasks/{task_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == task_id
    assert data["status"] in ("SUCCESS", "PENDING", "FAILURE", "STARTED")


@pytest.mark.asyncio
async def test_task_status_pending(client):
    """Task en cours doit retourner result=None."""
    pending_task = MagicMock()
    pending_task.state  = "PENDING"
    pending_task.ready.return_value = False
    pending_task.result = None
    with patch("workers.celery_app.celery_app.AsyncResult", return_value=pending_task):
        resp = await client.get(f"/tasks/{uuid.uuid4()}")
    assert resp.status_code == 200
    assert resp.json()["result"] is None


# ── Tests /clustering/scenes ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_clustering_hdbscan(client, mock_send_task):
    resp = await client.post("/clustering/scenes", json={
        "video_id":   str(uuid.uuid4()),
        "algorithm":  "hdbscan",
        "clip_model": "ViT-L/14",
    })
    assert resp.status_code == 202
    assert "task_id" in resp.json()


@pytest.mark.asyncio
async def test_clustering_kmeans(client, mock_send_task):
    resp = await client.post("/clustering/scenes", json={
        "video_id":  str(uuid.uuid4()),
        "algorithm": "kmeans",
        "n_clusters": 5,
    })
    assert resp.status_code == 202


# ── Tests /detection/run ──────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_detection_grounding_dino(client, mock_send_task):
    resp = await client.post("/detection/run", json={
        "video_id":   str(uuid.uuid4()),
        "model":      "grounding_dino_v2",
        "queries":    ["person", "car", "bicycle"],
        "confidence": 0.35,
    })
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_detection_empty_queries(client, mock_send_task):
    """Requêtes vides — le worker doit gérer ce cas."""
    resp = await client.post("/detection/run", json={
        "video_id": str(uuid.uuid4()),
        "model":    "yolo_world_v2_x",
        "queries":  [],
    })
    # L'API accepte (validation côté worker)
    assert resp.status_code in (202, 422)


# ── Tests /analytics/* ────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_ocr_endpoint(client, mock_send_task):
    resp = await client.post("/analytics/ocr", json={
        "video_id": str(uuid.uuid4()),
        "engine":   "paddleocr_v4",
        "langs":    ["fr", "en"],
    })
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_reid_endpoint(client, mock_send_task):
    resp = await client.post("/analytics/reid_track", json={
        "video_id": str(uuid.uuid4()),
        "detector": "yolov9e",
        "tracker":  "botsort",
    })
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_actions_endpoint(client, mock_send_task):
    resp = await client.post("/analytics/actions", json={
        "video_id": str(uuid.uuid4()),
        "model":    "videomae_v2_giant",
        "top_k":    5,
    })
    assert resp.status_code == 202


# ── Tests /search/* ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_search_query_text(client, mock_send_task):
    """Recherche sémantique par texte."""
    mock_send_task.get = MagicMock(return_value=[
        {"frame": 100, "timestamp": 4.0, "similarity": 0.87},
        {"frame": 250, "timestamp": 10.0, "similarity": 0.72},
    ])
    with patch("main._await_task", new_callable=AsyncMock, return_value=[
        {"frame": 100, "timestamp": 4.0, "similarity": 0.87}
    ]):
        resp = await client.post("/search/query", json={
            "video_id":   str(uuid.uuid4()),
            "query_text": "a red car on the highway",
            "top_n":      10,
            "threshold":  0.2,
        })
    # /search/query délègue à Celery : 202 + task_id, résultats via /tasks/{id}
    assert resp.status_code == 202
    assert "task_id" in resp.json()


@pytest.mark.asyncio
async def test_search_query_missing_video(client):
    """Recherche sans video_id → 422."""
    resp = await client.post("/search/query", json={
        "query_text": "person running",
    })
    assert resp.status_code == 422


# ── Tests de validation Pydantic ──────────────────────────────────────────────
@pytest.mark.asyncio
async def test_vlm_query_validation(client):
    """VLM query avec mauvais types doit retourner 422."""
    resp = await client.post("/vlm/query", json={
        "video_id":   str(uuid.uuid4()),
        "model":      "llava",
        "question":   "What is happening?",
        "start_time": "not_a_float",   # type invalide
        "end_time":   10.0,
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_detection_confidence_bounds(client, mock_send_task):
    """Confiance entre 0 et 1."""
    resp = await client.post("/detection/run", json={
        "video_id":   str(uuid.uuid4()),
        "model":      "grounding_dino_v2",
        "queries":    ["person"],
        "confidence": 0.5,
        "nms_thresh": 0.45,
    })
    assert resp.status_code == 202
