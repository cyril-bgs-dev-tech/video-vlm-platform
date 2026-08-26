"""
api/routers/upload.py — Router upload vidéo
Validation format, extraction métadonnées (ffprobe), stockage MinIO optionnel
"""
import os
import uuid
import subprocess
import json
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

VIDEO_DIR  = Path(os.getenv("VIDEO_DIR", "/data/videos"))
MAX_SIZE   = int(os.getenv("MAX_VIDEO_SIZE_MB", 2048)) * 1024 * 1024  # bytes

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".ts"}
ALLOWED_MIMETYPES  = {
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/webm", "video/mp2t",
}


class VideoMetadata(BaseModel):
    video_id:     str
    filename:     str
    path:         str
    size_bytes:   int
    duration_sec: Optional[float] = None
    fps:          Optional[float] = None
    width:        Optional[int]   = None
    height:       Optional[int]   = None
    codec:        Optional[str]   = None
    format_name:  Optional[str]   = None


def _probe_video(path: str) -> dict:
    """Extrait les métadonnées via ffprobe (si disponible)."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)

        # Chercher le premier stream vidéo
        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            {}
        )
        fmt = data.get("format", {})

        # Calculer le FPS depuis r_frame_rate (ex: "25/1")
        fps = None
        if "r_frame_rate" in video_stream:
            parts = video_stream["r_frame_rate"].split("/")
            if len(parts) == 2 and int(parts[1]) > 0:
                fps = round(int(parts[0]) / int(parts[1]), 3)

        return {
            "duration_sec": float(fmt.get("duration", 0)) or None,
            "fps":          fps,
            "width":        video_stream.get("width"),
            "height":       video_stream.get("height"),
            "codec":        video_stream.get("codec_name"),
            "format_name":  fmt.get("format_name", "").split(",")[0],
        }
    except Exception as e:
        logger.warning("ffprobe failed", path=str(path), error=str(e))
        return {}


@router.post("", response_model=VideoMetadata)
async def upload_video(
    file: UploadFile = File(...),
    store_minio: bool = Query(False, description="Copier aussi vers MinIO"),
):
    """
    Upload une vidéo et retourne ses métadonnées.

    - Valide l'extension et le type MIME
    - Streame vers le disque en chunks de 8 Mo (évite OOM)
    - Extrait les métadonnées via ffprobe
    - Optionnellement réplique vers MinIO
    """
    # ── Validation ────────────────────────────────────────────
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Extension {ext!r} non supportée. Formats acceptés : {sorted(ALLOWED_EXTENSIONS)}",
        )

    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type and content_type not in ALLOWED_MIMETYPES:
        logger.warning("Type MIME suspect", content_type=content_type, original_name=file.filename)
        # Warning seulement — certains navigateurs envoient des types génériques

    # ── Stockage ──────────────────────────────────────────────
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_id = str(uuid.uuid4())
    dest     = VIDEO_DIR / f"{video_id}{ext}"

    size_bytes = 0
    async with aiofiles.open(dest, "wb") as f:
        while True:
            chunk = await file.read(8 * 1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > MAX_SIZE:
                # Supprimer le fichier partiel
                await f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Fichier trop grand : limite {MAX_SIZE // 1024 // 1024} Mo",
                )
            await f.write(chunk)

    logger.info(
        "Video uploaded",
        video_id=video_id,
        video_filename=file.filename,
        size_mb=round(size_bytes / 1024 / 1024, 2),
    )

    # ── Métadonnées ───────────────────────────────────────────
    meta = _probe_video(str(dest))

    # ── MinIO (optionnel) ─────────────────────────────────────
    if store_minio:
        try:
            from minio import Minio
            client = Minio(
                os.getenv("MINIO_ENDPOINT", "minio:9000"),
                access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
                secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
                secure=False,
            )
            bucket = os.getenv("MINIO_BUCKET", "videos")
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
            client.fput_object(bucket, f"{video_id}{ext}", str(dest))
            logger.info("Video stored in MinIO", video_id=video_id, bucket=bucket)
        except Exception as e:
            logger.warning("MinIO upload failed", video_id=video_id, error=str(e))

    return VideoMetadata(
        video_id=video_id,
        filename=file.filename or f"{video_id}{ext}",
        path=str(dest),
        size_bytes=size_bytes,
        **meta,
    )


@router.get("/list")
async def list_videos(limit: int = Query(50, le=200)):
    """Liste les vidéos stockées sur le disque."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    videos = []
    for p in sorted(VIDEO_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        if p.suffix.lower() in ALLOWED_EXTENSIONS:
            stat = p.stat()
            videos.append({
                "video_id": p.stem,
                "filename": p.name,
                "size_mb":  round(stat.st_size / 1024 / 1024, 2),
                "mtime":    stat.st_mtime,
            })
    return {"videos": videos, "total": len(videos)}


@router.delete("/{video_id}")
async def delete_video(video_id: str):
    """Supprime une vidéo du disque."""
    matches = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Vidéo {video_id!r} introuvable")
    for p in matches:
        p.unlink()
        logger.info("Video deleted", video_id=video_id, path=str(p))
    return {"deleted": video_id, "files_removed": len(matches)}
