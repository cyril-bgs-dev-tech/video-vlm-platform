"""
streamlit_app/utils/video_utils.py — Utilitaires vidéo partagés
Extraction frames, affichage overlay, conversion timecodes
"""
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Timecodes
# ─────────────────────────────────────────────────────────────────────────────
def seconds_to_tc(seconds: float) -> str:
    """Convertit un float en timecode HH:MM:SS.mmm"""
    h   = int(seconds // 3600)
    m   = int((seconds % 3600) // 60)
    s   = int(seconds % 60)
    ms  = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def tc_to_seconds(tc: str) -> float:
    """Convertit HH:MM:SS ou MM:SS en secondes float."""
    parts = tc.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def frame_to_seconds(frame_idx: int, fps: float) -> float:
    return round(frame_idx / fps, 3) if fps > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Extraction de frames
# ─────────────────────────────────────────────────────────────────────────────
def get_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """
    Extrait une frame précise d'une vidéo.
    Retourne un tableau RGB numpy ou None si la lecture échoue.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def get_video_info(video_path: str) -> dict:
    """Retourne les métadonnées d'une vidéo via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"]
    cap.release()
    return info


def extract_thumbnails(
    video_path: str,
    n: int = 8,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> list[np.ndarray]:
    """Extrait N thumbnails uniformément répartis sur [start_sec, end_sec]."""
    info = get_video_info(video_path)
    fps  = info["fps"]
    total = info["total_frames"]

    end_frame = int((end_sec or info["duration_sec"]) * fps)
    end_frame = min(end_frame, total - 1)
    start_frame = int(start_sec * fps)

    if end_frame <= start_frame or n <= 0:
        return []

    indices = [
        start_frame + int(i * (end_frame - start_frame) / (n - 1))
        for i in range(n)
    ]

    cap    = cv2.VideoCapture(video_path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Overlay des détections sur une frame
# ─────────────────────────────────────────────────────────────────────────────
COLORS = [
    (0, 200, 100),   # vert
    (66, 135, 245),  # bleu
    (245, 130, 48),  # orange
    (240, 50, 230),  # violet
    (70, 240, 240),  # cyan
    (255, 225, 25),  # jaune
    (230, 25, 75),   # rouge
    (145, 30, 180),  # pourpre
]


def draw_detections(
    frame_rgb: np.ndarray,
    detections: list[dict],
    show_label: bool = True,
    show_score: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """
    Dessine les bounding boxes et labels sur une frame RGB.

    detections : liste de dicts avec au moins {"bbox": [x1,y1,x2,y2]}
                 optionnellement {"class": "...", "confidence": 0.xx, "track_id": N}
    """
    frame = frame_rgb.copy()
    classes_seen: dict[str, int] = {}

    for det in detections:
        bbox = det.get("bbox") or det.get("box")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

        cls  = det.get("class", det.get("label", "obj"))
        conf = det.get("confidence", det.get("score", 0.0))
        tid  = det.get("track_id")

        # Couleur stable par classe
        if cls not in classes_seen:
            classes_seen[cls] = len(classes_seen) % len(COLORS)
        color = COLORS[classes_seen[cls]]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if show_label or show_score:
            parts = []
            if tid is not None:
                parts.append(f"#{tid}")
            if show_label:
                parts.append(cls)
            if show_score:
                parts.append(f"{conf:.2f}")
            text = " ".join(parts)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 4, th + 4)
            cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
            cv2.putText(frame, text, (x1 + 2, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_tracks(
    frame_rgb: np.ndarray,
    tracks: list[dict],
    history: Optional[dict] = None,
) -> np.ndarray:
    """
    Dessine les tracks avec trajectoires optionnelles.
    history : {track_id: [(x_center, y_center), ...]}
    """
    frame = draw_detections(frame_rgb, tracks)
    if history:
        for tid, points in history.items():
            if len(points) < 2:
                continue
            color = COLORS[int(tid) % len(COLORS)]
            pts   = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Affichage Streamlit
# ─────────────────────────────────────────────────────────────────────────────
def show_frame_with_detections(
    video_path: str,
    frame_idx: int,
    detections: Optional[list[dict]] = None,
    caption: Optional[str] = None,
) -> None:
    """Widget Streamlit : frame avec overlay des détections."""
    frame = get_frame(video_path, frame_idx)
    if frame is None:
        st.warning(f"Impossible de lire la frame {frame_idx}")
        return

    if detections:
        frame = draw_detections(frame, detections)

    info = get_video_info(video_path)
    tc   = seconds_to_tc(frame_to_seconds(frame_idx, info["fps"]))

    st.image(
        frame,
        caption=caption or f"Frame {frame_idx}  ·  {tc}",
        use_container_width=True,
    )


def show_thumbnail_strip(
    video_path: str,
    n: int = 6,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> None:
    """Bande de N thumbnails répartis sur la durée sélectionnée."""
    thumbnails = extract_thumbnails(video_path, n, start_sec, end_sec)
    if not thumbnails:
        st.info("Aucune thumbnail disponible")
        return

    info = get_video_info(video_path)
    fps  = info["fps"]
    end  = end_sec or info["duration_sec"]
    step = (end - start_sec) / max(len(thumbnails) - 1, 1)

    cols = st.columns(len(thumbnails))
    for i, (col, img) in enumerate(zip(cols, thumbnails)):
        t = start_sec + i * step
        with col:
            st.image(img, use_container_width=True)
            st.caption(seconds_to_tc(t))
