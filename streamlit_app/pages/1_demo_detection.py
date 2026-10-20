"""
Page 1 — Démo détection vidéo en direct
Clic Analyser → la vidéo défile avec bbox + labels + scores en temps réel (WebSocket).
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Queue

import cv2
import httpx
import numpy as np
import pandas as pd
import streamlit as st

from utils.ui_components import inject_global_css, render_section_header
from utils.video_utils import draw_detections, get_frame, get_video_info

inject_global_css()

API = (
    os.getenv("API_BASE_URL")
    or st.secrets.get("API_BASE_URL", "http://localhost:8000")
).rstrip("/")
_ws_default = API.replace("https://", "wss://").replace("http://", "ws://")
WS_BASE = (
    os.getenv("WS_BASE_URL")
    or st.secrets.get("WS_BASE_URL", _ws_default)
).rstrip("/")

MODELS = {
    "YOLO-World v2 (rapide / live)": "yolo_world_v2",
    "GroundingDINO (précis / live)": "grounding_dino",
    "Mini-InternVL-4B (VLM · ~10-15 s/frame)": "mini_internvl_4b",
    "InternVL3.5-8B (VLM · ~10-15 s/frame)": "internvl35_8b",
    "LocateAnything-3B (en cours d'entretien)": "locate_anything_3b",
    "YOLO VisDrone (aérien · 10 classes fixes)": "yolo_visdrone_s",
}

# Modèles en cours d'entretien : affichés, mais non lançables tant que leur détection ne fonctionne pas.
EN_ENTRETIEN = {
    "locate_anything_3b": "son code distant est incompatible avec transformers v5 : le modèle charge et génère, mais sa sortie est dégénérée et ne produit aucune boîte",
}
MODELES_ACTIFS = [label for label, mid in MODELS.items() if mid not in EN_ENTRETIEN]

DEMO_VIDEOS = [
    {"name": "Trafic urbain dense", "file": "urban_traffic.mp4",
     "queries": ["car", "truck", "bus", "person", "traffic light", "motorcycle", "sign"],
     "icon": "🚗", "specialty": "véhicules & transport",
     "description": "Carrefour urbain très fréquenté — voitures, bus, deux-roues, piétons et feux tricolores"},
    {"name": "Piétons & foule", "file": "street_pedestrians.mp4",
     "queries": ["person", "backpack", "handbag", "umbrella", "stroller", "dog"],
     "icon": "🚶", "specialty": "personnes & actions",
     "description": "Passage piéton animé — personnes seules, groupes, accessoires et animaux"},
    {"name": "Rue commerçante", "file": "bustling_street.mp4",
     "queries": ["person", "car", "bicycle", "bus", "traffic light", "sign", "storefront"],
     "icon": "🏙️", "specialty": "scène urbaine complète",
     "description": "Avenue commerçante — piétons, vitrines, véhicules, panneaux et mobilité douce"},
    {"name": "Vue aérienne drone", "file": "aerial_drone_view.mp4",
     "queries": ["car", "vehicle", "person", "building", "traffic light", "bus", "motorcycle"],
     "icon": "🛩️", "specialty": "géospatial & drone",
     "description": "Centre-ville vu du drone — voitures, piétons, buildings et traffic entre les gratte-ciel"},
    {"name": "Faune & animaux", "file": "wildlife_safari.mp4",
     "queries": ["elephant", "giraffe", "zebra", "bird", "tree", "grass", "animal"],
     "icon": "🦁", "specialty": "animaux & biodiversité",
     "description": "Savane africaine — faune sauvage dans son habitat naturel, grands mammifères et oiseaux"},
    {"name": "Sports collectifs", "file": "team_sports_match.mp4",
     "queries": ["person", "ball", "player", "goal", "net", "crowd", "uniform"],
     "icon": "⚽", "specialty": "sports & action",
     "description": "Match de sport collectif — joueurs en mouvement, ballon, but et public"},
    {"name": "Conduite de nuit", "file": "night_drive.mp4",
     "queries": ["car", "traffic light", "sign", "person", "street light", "building", "headlight"],
     "icon": "🌙", "specialty": "conduite & faible luminosité",
     "description": "Trafic nocturne — véhicules, éclairage public, panneaux et piétons sous faible lumière"},
    {"name": "Marché & étals", "file": "market_stalls.mp4",
     "queries": ["person", "fruit", "vegetable", "box", "table", "bottle", "canopy"],
     "icon": "🧺", "specialty": "retail & marchés",
     "description": "Marché de plein air — étals, produits frais, clients et marchandises diverses"},
    {"name": "Chantier BTP", "file": "construction_site.mp4",
     "queries": ["person", "truck", "crane", "hard hat", "tool", "concrete", "excavator"],
     "icon": "🏗️", "specialty": "industrie & construction",
     "description": "Chantier de construction — engins, ouvriers, équipements de sécurité et matériaux"},
    {"name": "Sous-marin & plongée", "file": "underwater_dive.mp4",
     "queries": ["fish", "coral", "diver", "turtle", "jellyfish", "reef", "sea"],
     "icon": "🐠", "specialty": "sous-marin & océan",
     "description": "Fond marin tropical — poissons, coraux, plongeur et vie aquatique colorée"},
]

_defaults = {
    "video_id": None,
    "video_path": None,
    "video_name": None,
    "api_upload_ok": False,
    "detection_queries": ["person", "car"],
    "detection_results": None,
    "detection_error": None,
    "model_eval": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _sample_path(filename: str) -> Path | None:
    for root in (
        Path("/app/sample_videos"),
        Path(__file__).resolve().parent.parent / "sample_videos",
        Path(__file__).resolve().parent.parent.parent / "sample_videos",
        Path("sample_videos"),
    ):
        p = root / filename
        if p.exists():
            return p
    return None


def _upload(path: str, name: str) -> tuple[bool, str]:
    try:
        with open(path, "rb") as f:
            r = httpx.post(
                f"{API}/upload",
                files={"file": (name, f, "video/mp4")},
                timeout=180.0,
            )
        if r.status_code == 200:
            return True, r.json()["video_id"]
        return False, f"HTTP {r.status_code}: {r.text[:300]}"
    except Exception as e:
        return False, str(e)


def _start_batch(video_id: str, model: str, queries: list, confidence: float) -> tuple[bool, str]:
    try:
        r = httpx.post(
            f"{API}/detection/run",
            json={
                "video_id": video_id,
                "model": model,
                "queries": queries,
                "confidence": confidence,
                "track": False,
            },
            timeout=30.0,
        )
        if r.status_code not in (200, 202):
            return False, f"HTTP {r.status_code}: {r.text[:300]}"
        tid = r.json().get("task_id")
        return (True, tid) if tid else (False, r.text[:200])
    except Exception as e:
        return False, str(e)


def _poll_task(task_id: str, timeout: float = 600.0) -> dict:
    bar = st.progress(0.0, text="Analyse batch…")
    t0 = time.time()
    while time.time() - t0 < timeout:
        bar.progress(min(0.95, (time.time() - t0) / 120), text=f"Batch… {time.time()-t0:.0f}s")
        try:
            r = httpx.get(f"{API}/tasks/{task_id}", timeout=15.0)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "SUCCESS":
                    bar.progress(1.0, text="OK")
                    return data.get("result") or {}
                if data.get("status") in ("FAILURE", "REVOKED"):
                    return {"error": data.get("error") or data.get("status")}
        except Exception:
            pass
        time.sleep(0.8)
    return {"error": "Timeout batch"}


def _normalize_rows(raw_list: list) -> list[dict]:
    rows = []
    for d in raw_list or []:
        if not isinstance(d, dict):
            continue
        label = d.get("class") or d.get("label") or "object"
        score = float(d.get("confidence", d.get("score", 0)) or 0)
        bbox = d.get("bbox") or d.get("box") or [0, 0, 0, 0]
        rows.append({
            "frame": int(d.get("frame", 0)),
            "timestamp": float(d.get("timestamp", 0) or 0),
            "class": str(label),
            "label": str(label),
            "confidence": round(score, 3),
            "score_pct": f"{score * 100:.1f}%",
            "bbox": [int(x) for x in list(bbox)[:4]],
        })
    return rows


def _ws_receiver(
    video_id: str,
    model: str,
    queries: list,
    fps: int,
    confidence: float,
    q: Queue,
    pace: str = "realtime",
):
    """
    Thread WS → queue UI.
    Backpressure : si l'UI est en retard, on ne garde que le dernier message
    de type detection (drop-old) pour rester fluide.
    """

    async def _run():
        uri = f"{WS_BASE}/ws/detect/{video_id}"
        try:
            import websockets
            async with websockets.connect(
                uri,
                open_timeout=25,
                ping_interval=20,
                max_size=12_000_000,
                compression=None,
            ) as ws:
                await ws.send(json.dumps({
                    "model": model,
                    "query": ", ".join(queries),
                    "queries": queries,
                    "fps": int(fps),
                    "confidence": float(confidence),
                    "send_jpeg": True,
                    "max_width": 960,
                    "jpeg_quality": 68,
                    "pace": pace,
                }))
                async for raw in ws:
                    data = json.loads(raw)
                    # drop-old non bloquant : si file pleine, jeter les plus vieux detections
                    if data.get("type") == "detection":
                        while q.full():
                            try:
                                old = q.get_nowait()
                                if old.get("type") in ("done", "error", "started"):
                                    # conserver messages critiques
                                    try:
                                        q.put_nowait(old)
                                    except Exception:
                                        pass
                                    break
                            except Empty:
                                break
                    try:
                        q.put_nowait(data)
                    except Exception:
                        # file saturée : drop ce frame
                        pass
                    if data.get("type") in ("done", "error"):
                        break
        except Exception as e:
            try:
                q.put_nowait({"type": "error", "message": str(e)})
            except Exception:
                pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


def _jpeg_b64_to_rgb(b64: str):
    import base64
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_frame_rgb(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _norm_det(d: dict, frame_idx: int, ts: float) -> dict:
    label = d.get("class") or d.get("label") or d.get("c") or "object"
    score = float(d.get("confidence", d.get("score", d.get("s", 0))) or 0)
    bbox = d.get("bbox") or d.get("box") or d.get("b") or [0, 0, 0, 0]
    return {
        "frame": frame_idx,
        "timestamp": ts,
        "class": str(label),
        "label": str(label),
        "confidence": round(score, 3),
        "score_pct": f"{score * 100:.1f}%",
        "bbox": [int(x) for x in list(bbox)[:4]],
    }


def _run_live_analysis(
    video_path: str,
    video_id: str,
    model: str,
    queries: list,
    confidence: float,
    analysis_fps: int,
    video_info: dict,
    pace: str = "realtime",
    header: str | None = None,
) -> dict:
    """
    Flux UI optimisé :
      WS (JPEG annoté + dets) → drop-old queue → affichage immédiat
    Pas de seek OpenCV si jpeg présent = fluide.
    """
    st.markdown("---")
    render_section_header("🔴", header or "Analyse en direct", "Flux optimisé · bbox + labels + scores")

    ph_img = st.empty()
    ph_meta = st.empty()
    ph_table = st.empty()
    ph_bar = st.progress(0.0, text="Connexion WebSocket…")
    ph_status = st.empty()

    q: Queue = Queue(maxsize=8)
    thr = threading.Thread(
        target=_ws_receiver,
        args=(video_id, model, queries, analysis_fps, confidence, q, pace),
        daemon=True,
    )
    thr.start()

    # fallback local uniquement si pas de jpeg
    cap = None
    vfps = float(video_info.get("fps") or 25.0)
    total = int(video_info.get("total_frames") or 0)

    all_rows: list[dict] = []
    infer_samples: list[float] = []
    is_mock = False
    model_name = model
    device = "?"
    err = None
    n_frames_ui = 0
    t_start = time.time()
    deadline = t_start + 900
    last_ui = 0.0
    min_ui_interval = 1.0 / 30.0  # plafond refresh UI ~30 Hz

    ph_status.info("Connexion au worker… première frame dès le warmup modèle.")

    while time.time() < deadline:
        try:
            msg = q.get(timeout=0.25)
        except Empty:
            if not thr.is_alive() and q.empty():
                if not all_rows and not err:
                    err = "WebSocket fermé sans résultats"
                break
            continue

        mtype = msg.get("type")

        if mtype == "started":
            is_mock = bool(msg.get("mock"))
            model_name = msg.get("model") or model
            device = msg.get("device") or device
            mode = "JPEG stream" if msg.get("send_jpeg") else "bbox only"
            if is_mock:
                ph_status.warning("Mode mock (modèle non chargé)")
            else:
                ph_status.success(
                    f"Live **{model_name}** · {device} · {mode} · pace={msg.get('pace')} · "
                    f"{msg.get('fps')} FPS analyse"
                )
            continue

        if mtype == "error":
            err = msg.get("message") or "Erreur stream"
            ph_status.error(err)
            break

        if mtype == "done":
            ph_bar.progress(1.0, text=f"Terminé · {n_frames_ui} frames UI · {len(all_rows)} dets")
            break

        if mtype != "detection":
            continue

        # drain : garder le message le plus récent si backlog
        latest = msg
        try:
            while True:
                nxt = q.get_nowait()
                if nxt.get("type") == "detection":
                    latest = nxt
                elif nxt.get("type") in ("done", "error", "started"):
                    q.put(nxt)
                    break
                else:
                    latest = nxt
        except Empty:
            pass
        msg = latest

        frame_idx = int(msg.get("frame", 0))
        ts = float(msg.get("timestamp", frame_idx / max(vfps, 1e-6)))
        dets = msg.get("detections") or []
        if msg.get("infer_ms") is not None:
            try:
                infer_samples.append(float(msg["infer_ms"]))
            except (TypeError, ValueError):
                pass
        for d in dets:
            all_rows.append(_norm_det(d, frame_idx, ts))

        # throttle UI pour éviter de saturer Streamlit
        now = time.time()
        if now - last_ui < min_ui_interval and not msg.get("jpeg"):
            continue
        last_ui = now

        labels = " · ".join(
            f"{(d.get('class') or d.get('c') or d.get('label'))} "
            f"{float(d.get('confidence', d.get('s', d.get('score', 0)))):.0%}"
            for d in dets[:6]
        ) or "—"

        # ── Chemin fluide : JPEG déjà annoté côté worker ──
        shown = False
        if msg.get("jpeg"):
            rgb = _jpeg_b64_to_rgb(msg["jpeg"])
            if rgb is not None:
                ph_img.image(
                    rgb,
                    use_container_width=True,
                    caption=f"▶ {ts:.2f}s · f{frame_idx} · {len(dets)} obj · {labels}",
                )
                shown = True
                n_frames_ui += 1

        # ── Fallback local seek (plus lent) ──
        if not shown and video_path:
            if cap is None:
                cap = cv2.VideoCapture(video_path)
            rgb = _read_frame_rgb(cap, frame_idx)
            if rgb is not None:
                vis = draw_detections(rgb, [_norm_det(d, frame_idx, ts) for d in dets])
                ph_img.image(
                    vis,
                    use_container_width=True,
                    caption=f"▶ {ts:.2f}s · f{frame_idx} · {len(dets)} obj · {labels}",
                )
                n_frames_ui += 1

        prog = msg.get("progress")
        if prog is None and total:
            prog = frame_idx / total
        lag_note = ""
        if msg.get("overload"):
            lag_note = " · GPU overload→skip"
        if msg.get("skipped"):
            lag_note += f" · skip={msg.get('skipped')}"

        ph_bar.progress(
            min(max(float(prog or 0), 0.0), 0.99),
            text=f"{ts:.1f}s · {len(all_rows)} dets · {msg.get('infer_ms', '—')} ms{lag_note}",
        )
        ph_meta.markdown(
            f"| t | frame | dets | infer | UI frames |\n|---:|---:|---:|---:|---:|\n"
            f"| **{ts:.2f}s** | **{frame_idx}** | **{len(all_rows)}** | "
            f"{msg.get('infer_ms', '—')} ms | **{n_frames_ui}** |"
        )
        # table throttled (toutes les 5 frames UI)
        if all_rows and n_frames_ui % 3 == 0:
            ph_table.dataframe(
                pd.DataFrame(all_rows).tail(10)[
                    ["frame", "timestamp", "class", "confidence", "score_pct"]
                ].iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )

    if cap is not None:
        cap.release()

    wall_s = max(time.time() - t_start, 1e-6)

    if err and not all_rows:
        return {"error": err}

    return {
        "results": all_rows,
        "model": model_name,
        "mock": is_mock,
        "status": "mock" if is_mock else "ok",
        "device": device,
        "stream_error": err,
        "n_detections": len(all_rows),
        "ui_frames": n_frames_ui,
        "wall_s": round(wall_s, 2),
        "eff_fps": round(n_frames_ui / wall_s, 2),
        "avg_infer_ms": round(sum(infer_samples) / len(infer_samples), 1) if infer_samples else None,
        "p95_infer_ms": round(sorted(infer_samples)[int(len(infer_samples) * 0.95)], 1)
        if len(infer_samples) >= 3 else None,
    }


def _replay_with_results(video_path: str, rows: list[dict], video_info: dict, speed: float = 1.0):
    """
    Rejoue la vidéo annotée à vitesse réelle.
    Décodage séquentiel (grab/retrieve) — pas de seek OpenCV = replay fluide.
    """
    if not rows or not video_path:
        return
    df = pd.DataFrame(rows)
    if "frame" not in df.columns:
        return
    frames = sorted(df["frame"].unique().tolist())
    vfps = float(video_info.get("fps") or 25.0)

    ph = st.empty()
    cap = cv2.VideoCapture(video_path)
    by_f = {int(f): g.to_dict("records") for f, g in df.groupby("frame")}

    fmax = int(frames[-1])
    step = max(1, int(round(np.median(np.diff(frames))))) if len(frames) > 1 else max(1, int(vfps // 2))
    frame_delay = (step / max(vfps, 1)) / max(speed, 0.1)

    last_dets: list[dict] = []
    fi = 0
    t_next = time.time()
    while fi <= fmax:
        if not cap.grab():
            break
        if fi % step:
            fi += 1
            continue

        if fi in by_f:
            last_dets = by_f[fi]
        else:
            nearer = min(frames, key=lambda x: abs(int(x) - fi))
            if abs(int(nearer) - fi) <= step:
                last_dets = by_f[int(nearer)]

        ok, frame = cap.retrieve()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vis = draw_detections(rgb, last_dets, show_label=True, show_score=True)
            ph.image(
                vis,
                use_container_width=True,
                caption=f"▶ Replay {fi / vfps:.2f}s · {len(last_dets)} obj",
            )

        # cadence : caler sur l'horloge plutôt qu'un sleep fixe (absorbe le temps de rendu)
        t_next += frame_delay
        pause = t_next - time.time()
        if pause > 0.002:
            time.sleep(pause)
        fi += 1
    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
st.title("🎬 Démo détection — live")
st.caption(
    f"API `{API}` · WS `{WS_BASE}` — au clic, la vidéo défile avec **bbox + label + score**."
)

@st.cache_data(ttl=15, show_spinner=False)
def _api_health() -> bool:
    # /health interroge Celery (inspect ~2 s) : timeout large + cache 15 s
    try:
        return httpx.get(f"{API}/health", timeout=8.0).status_code == 200
    except Exception:
        return False

if _api_health():
    st.success("API connectée")
else:
    st.error("API injoignable")

# ── 1. Vidéo ──────────────────────────────────────────────────────────────────
render_section_header("📤", "1. Charger une vidéo", "")
t_up, t_demo = st.tabs(["Upload", "Démos"])

with t_up:
    up = st.file_uploader("Fichier vidéo", type=["mp4", "mov", "avi", "mkv", "webm"])
    if up is not None and st.button("Envoyer vers l'API", type="primary"):
        suffix = Path(up.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(up.getbuffer())
        tmp.close()
        st.session_state.video_path = tmp.name
        st.session_state.video_name = up.name
        st.session_state.detection_results = None
        st.session_state.detection_error = None
        with st.spinner("Upload…"):
            ok, val = _upload(tmp.name, up.name)
        if ok:
            st.session_state.video_id = val
            st.session_state.api_upload_ok = True
            st.success(f"OK `{val[:12]}…`")
        else:
            st.session_state.api_upload_ok = False
            st.error(val)
        st.rerun()

with t_demo:
    @st.cache_data(ttl=30)
    def _fetch_demo_datasets():
        try:
            r = httpx.get(f"{API}/demo-datasets", timeout=5.0)
            if r.status_code == 200:
                return r.json().get("datasets", DEMO_VIDEOS)
        except Exception:
            pass
        return DEMO_VIDEOS
    _demo_list = _fetch_demo_datasets()

    for row_idx in range(0, len(_demo_list), 5):
        row = _demo_list[row_idx:row_idx + 5]
        cols = st.columns(len(row))
        for j, col in enumerate(cols):
            i = row_idx + j
            demo = row[j]
            with col:
                st.markdown(f"**{demo['icon']} {demo['name']}**")
                st.caption(demo.get('specialty', ''))
                qs = demo.get("queries", [])
                if qs:
                    st.markdown(
                        " " + " ".join(f"<code>{q}</code>" for q in qs[:6]),
                        unsafe_allow_html=True,
                    )
                exists = _sample_path(demo["file"]) is not None
                btn_label = "Charger" if exists else "📥 Vidéo manquante"
                if st.button(btn_label, key=f"demo_{i}", disabled=not exists, use_container_width=True):
                    import shutil
                    src = _sample_path(demo["file"])
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=src.suffix)
                    shutil.copy(src, tmp.name)
                    tmp.close()
                    st.session_state.video_path = tmp.name
                    st.session_state.video_name = demo["file"]
                    st.session_state.detection_queries = list(demo["queries"])
                    st.session_state.detection_results = None
                    with st.spinner("Upload…"):
                        ok, val = _upload(tmp.name, demo["file"])
                    st.session_state.api_upload_ok = ok
                    if ok:
                        st.session_state.video_id = val
                    else:
                        st.error(val)
                    st.rerun()

if not st.session_state.video_id:
    st.info("Chargez une vidéo pour continuer.")
    st.stop()

info = {}
if st.session_state.video_path:
    try:
        info = get_video_info(st.session_state.video_path)
    except Exception:
        pass

m1, m2, m3, m4 = st.columns(4)
m1.metric("Fichier", (st.session_state.video_name or "?")[:22])
m2.metric("Upload", "OK" if st.session_state.api_upload_ok else "KO")
m3.metric("Durée", f"{info.get('duration_sec', 0):.1f}s")
m4.metric("ID", f"{st.session_state.video_id[:10]}…")

# ── 2. Config ─────────────────────────────────────────────────────────────────
st.markdown("---")
render_section_header("🎯", "2. Classes, modèle & live", "")

pc = st.columns(3)
if pc[0].button("👤 Personnes", use_container_width=True):
    st.session_state.detection_queries = ["person", "backpack", "handbag"]
    st.rerun()
if pc[1].button("🚗 Véhicules", use_container_width=True):
    st.session_state.detection_queries = ["car", "truck", "bus", "motorcycle", "bicycle"]
    st.rerun()
if pc[2].button("🏙️ Urbain", use_container_width=True):
    st.session_state.detection_queries = ["person", "car", "bus", "traffic light", "bicycle"]
    st.rerun()

qtext = st.text_area(
    "Classes (une par ligne)",
    value="\n".join(st.session_state.detection_queries),
    height=90,
)
st.session_state.detection_queries = [q.strip() for q in qtext.splitlines() if q.strip()]

c1, c2, c3, c4 = st.columns(4)
with c1:
    model_label = st.selectbox(
        "Modèle", list(MODELS.keys()),
        format_func=lambda lbl: f"🔧 {lbl}" if MODELS[lbl] in EN_ENTRETIEN else lbl,
    )
    model_id = MODELS[model_label]
with c2:
    confidence = st.slider("Seuil", 0.1, 0.9, 0.30, 0.05)
with c3:
    analysis_fps = st.slider(
        "FPS analyse", 1, 15, 5,
        help="Cadence d'inférence. 5 = bon compromis fluidité / GPU.",
    )
with c4:
    live_mode = st.toggle("Temps réel (WS)", value=True)
    pace = st.selectbox(
        "Pace",
        ["realtime", "max"],
        format_func=lambda x: "Horloge vidéo (fluide)" if x == "realtime" else "Max GPU (plus rapide)",
        help="realtime = défilement calé sur le temps vidéo ; max = pousse dès que l'infer est prête",
    )
    can_run = st.session_state.api_upload_ok and bool(st.session_state.detection_queries)

if model_id in EN_ENTRETIEN:
    st.info(f"🔧 **{model_label.split(' (')[0]}** est en cours d'entretien : {EN_ENTRETIEN[model_id]}. "
            "Choisissez un autre modèle.")

run = st.button(
    "🔴 Analyser en direct" if live_mode else "🚀 Analyser (batch)",
    type="primary",
    disabled=not can_run or model_id in EN_ENTRETIEN,
    use_container_width=True,
)

if not st.session_state.api_upload_ok:
    st.warning("Upload API requis.")

# ── 2bis. Démo rapide — évaluation multi-modèles ──────────────────────────────
st.markdown("---")
render_section_header(
    "⚡", "Démo rapide — évaluation des modèles",
    "Chaque modèle analyse la vidéo en direct (lecture + bbox), puis comparatif automatique",
)

ec1, ec2 = st.columns([3, 1])
with ec1:
    eval_labels = st.multiselect(
        "Modèles à évaluer (passage séquentiel, aperçu live pour chacun)",
        MODELES_ACTIFS,
        default=MODELES_ACTIFS[:2],
        help="Les VLM (InternVL) sont précis mais lents (~10-15 s/frame) : "
             "l'aperçu live n'affichera que quelques frames. Ajoutez-les pour "
             "une évaluation complète.",
    )
with ec2:
    st.metric("Modèles sélectionnés", len(eval_labels))
_entretien = [lbl.split(" (")[0] for lbl, mid in MODELS.items() if mid in EN_ENTRETIEN]
if _entretien:
    st.caption("🔧 En cours d'entretien, donc non proposé ici : " + ", ".join(_entretien) + ".")

run_eval = st.button(
    "⚡ Lancer la démo rapide — évaluer les modèles",
    type="primary",
    disabled=not (can_run and eval_labels),
    use_container_width=True,
)

# ── 3. Run ────────────────────────────────────────────────────────────────────
if run:
    st.session_state.detection_results = None
    st.session_state.detection_error = None
    path = st.session_state.video_path
    vid = st.session_state.video_id
    queries = list(st.session_state.detection_queries)

    if live_mode:
        result = _run_live_analysis(
            path, vid, model_id, queries, confidence, analysis_fps, info, pace=pace,
        )
        # Fallback batch si WS KO
        if result.get("error") and not result.get("results"):
            st.warning(f"Live indisponible ({result['error']}) — bascule batch + replay…")
            ok, tid = _start_batch(vid, model_id, queries, confidence)
            if not ok:
                st.session_state.detection_error = tid
                st.error(tid)
            else:
                result = _poll_task(tid)
                if not result.get("error"):
                    rows = _normalize_rows(result.get("results") or [])
                    st.session_state.detection_results = {
                        "results": rows,
                        "model": result.get("model", model_id),
                        "mock": bool(result.get("mock")),
                        "device": result.get("device"),
                        "status": result.get("status", "ok"),
                    }
                    st.info("Replay vidéo annotée…")
                    _replay_with_results(path, rows, info, speed=1.0)
                    st.rerun()
                else:
                    st.session_state.detection_error = result.get("error")
                    st.error(result.get("error"))
        else:
            rows = result.get("results") or []
            st.session_state.detection_results = {
                "results": rows,
                "model": result.get("model", model_id),
                "mock": bool(result.get("mock")),
                "device": result.get("device"),
                "status": result.get("status", "ok"),
            }
            if result.get("stream_error"):
                st.warning(f"Stream: {result['stream_error']}")
            if rows:
                st.success(f"**{len(rows)}** détections live — voir le récap ci-dessous.")
            else:
                st.warning("Aucune détection.")
            st.rerun()
    else:
        ok, tid = _start_batch(vid, model_id, queries, confidence)
        if not ok:
            st.error(tid)
        else:
            result = _poll_task(tid)
            if result.get("error"):
                st.error(result["error"])
                st.session_state.detection_error = result["error"]
            else:
                rows = _normalize_rows(result.get("results") or [])
                st.session_state.detection_results = {
                    "results": rows,
                    "model": result.get("model", model_id),
                    "mock": bool(result.get("mock")),
                    "device": result.get("device"),
                    "status": result.get("status", "ok"),
                }
                st.info("Replay vidéo annotée…")
                _replay_with_results(path, rows, info, speed=1.0)
                st.rerun()

if run_eval:
    st.session_state.model_eval = None
    path = st.session_state.video_path
    vid = st.session_state.video_id
    queries = list(st.session_state.detection_queries)
    eval_results: dict[str, dict] = {}
    t_eval0 = time.time()
    eval_bar = st.progress(0.0, text="Évaluation multi-modèles…")

    for i, lbl in enumerate(eval_labels):
        mid = MODELS[lbl]
        eval_bar.progress(
            i / len(eval_labels),
            text=f"Modèle {i + 1}/{len(eval_labels)} — {lbl}",
        )
        result = _run_live_analysis(
            path, vid, mid, queries, confidence, analysis_fps, info,
            pace=pace, header=f"Évaluation {i + 1}/{len(eval_labels)} — {lbl}",
        )
        rows = result.get("results") or []
        df_m = pd.DataFrame(rows)
        eval_results[lbl] = {
            "model_id": mid,
            "error": result.get("error") or result.get("stream_error"),
            "mock": bool(result.get("mock")),
            "device": result.get("device"),
            "n_detections": len(rows),
            "n_classes": int(df_m["class"].nunique()) if "class" in df_m.columns and len(df_m) else 0,
            "avg_conf": float(df_m["confidence"].mean()) if "confidence" in df_m.columns and len(df_m) else 0.0,
            "avg_infer_ms": result.get("avg_infer_ms"),
            "p95_infer_ms": result.get("p95_infer_ms"),
            "eff_fps": result.get("eff_fps"),
            "wall_s": result.get("wall_s"),
            "ui_frames": result.get("ui_frames"),
            "rows": rows,
        }

    eval_bar.progress(
        1.0,
        text=f"Évaluation terminée · {len(eval_labels)} modèles · {time.time() - t_eval0:.0f}s",
    )
    st.session_state.model_eval = {
        "queries": queries,
        "confidence": confidence,
        "fps": analysis_fps,
        "video": st.session_state.video_name,
        "results": eval_results,
    }
    st.rerun()

if st.session_state.detection_error and not st.session_state.detection_results:
    st.error(st.session_state.detection_error)

# ── 3bis. Comparatif des modèles évalués ─────────────────────────────────────
ev = st.session_state.model_eval
if ev and ev.get("results"):
    st.markdown("---")
    render_section_header(
        "🏆", "Comparatif des modèles",
        f"Vidéo `{ev.get('video')}` · classes : {', '.join(ev.get('queries', []))} "
        f"· seuil {ev.get('confidence')} · {ev.get('fps')} FPS analyse",
    )

    er = ev["results"]
    ok_models = {k: v for k, v in er.items() if not v.get("error") or v.get("n_detections")}

    # Score démo (indicatif) : confiance 35% + vitesse 35% + couverture 30%
    scores: dict[str, float] = {}
    if ok_models:
        max_dets = max(v["n_detections"] for v in ok_models.values()) or 1
        infer_vals = [v["avg_infer_ms"] for v in ok_models.values() if v.get("avg_infer_ms")]
        min_infer = min(infer_vals) if infer_vals else None
        for k, v in ok_models.items():
            s_conf = v.get("avg_conf") or 0.0
            s_speed = (min_infer / v["avg_infer_ms"]) if (min_infer and v.get("avg_infer_ms")) else 0.5
            s_cov = v["n_detections"] / max_dets
            scores[k] = round(0.35 * s_conf + 0.35 * s_speed + 0.30 * s_cov, 3)

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if ranking:
        best_name = ranking[0][0]
        best = er[best_name]
        st.markdown(
            f"""<div class="model-eval-card winner">
            🥇 <strong>{best_name}</strong> — score démo <strong>{ranking[0][1]:.3f}</strong>
            · {best['n_detections']} détections · conf. moy {best['avg_conf']:.0%}
            · {best.get('avg_infer_ms') or '—'} ms/frame
            {'· <em>mode mock</em>' if best.get('mock') else ''}
            </div>""",
            unsafe_allow_html=True,
        )

    comp_rows = []
    for lbl, v in er.items():
        comp_rows.append({
            "Modèle": lbl,
            "Statut": "❌ " + str(v["error"])[:40] if (v.get("error") and not v["n_detections"])
            else ("🟡 mock" if v.get("mock") else "🟢 OK"),
            "Device": v.get("device") or "—",
            "Détections": v["n_detections"],
            "Classes": v["n_classes"],
            "Conf. moy": round(v["avg_conf"], 3),
            "Infer moy (ms)": v.get("avg_infer_ms"),
            "P95 (ms)": v.get("p95_infer_ms"),
            "FPS effectif": v.get("eff_fps"),
            "Durée (s)": v.get("wall_s"),
            "Score démo": scores.get(lbl),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df["Score démo"] = pd.to_numeric(comp_df["Score démo"], errors="coerce")
    comp_df = comp_df.sort_values("Score démo", ascending=False, na_position="last")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.caption(
        "Score démo (indicatif) = 35 % confiance moyenne + 35 % vitesse d'inférence "
        "+ 30 % couverture de détection — sur cette vidéo et ces classes uniquement."
    )

    tab_cmp, tab_replay_eval = st.tabs(["📈 Graphiques comparatifs", "🎞️ Replay par modèle"])

    with tab_cmp:
        try:
            import plotly.express as px
            plot_df = comp_df[comp_df["Détections"] > 0]
            if not plot_df.empty:
                cg1, cg2 = st.columns(2)
                with cg1:
                    fig_a = px.bar(
                        plot_df, x="Modèle", y="Détections", color="Modèle",
                        title="Détections par modèle",
                    )
                    fig_a.update_layout(template="plotly_dark", showlegend=False, height=320)
                    st.plotly_chart(fig_a, use_container_width=True)
                with cg2:
                    lat_df = plot_df.dropna(subset=["Infer moy (ms)"])
                    if not lat_df.empty:
                        fig_b = px.bar(
                            lat_df, x="Modèle", y="Infer moy (ms)", color="Modèle",
                            title="Latence d'inférence moyenne (ms)",
                        )
                        fig_b.update_layout(template="plotly_dark", showlegend=False, height=320)
                        st.plotly_chart(fig_b, use_container_width=True)
                fig_c = px.scatter(
                    plot_df.dropna(subset=["Infer moy (ms)"]),
                    x="Infer moy (ms)", y="Conf. moy", size="Détections",
                    color="Modèle", text="Modèle",
                    title="Compromis vitesse / confiance (taille = détections)",
                )
                fig_c.update_traces(textposition="top center")
                fig_c.update_layout(template="plotly_dark", height=380)
                st.plotly_chart(fig_c, use_container_width=True)
        except Exception as e:
            st.caption(str(e))

    with tab_replay_eval:
        replay_choices = [k for k, v in er.items() if v.get("rows")]
        if replay_choices:
            sel_model = st.selectbox("Modèle à rejouer", replay_choices, key="eval_replay_model")
            if st.button("▶️ Rejouer la vidéo annotée de ce modèle", use_container_width=True):
                _replay_with_results(
                    st.session_state.video_path, er[sel_model]["rows"], info, speed=1.0,
                )
        else:
            st.info("Aucune détection à rejouer.")

    if st.button("🧹 Effacer l'évaluation", use_container_width=True):
        st.session_state.model_eval = None
        st.rerun()

# ── 4. Récap ──────────────────────────────────────────────────────────────────
res = st.session_state.detection_results
if not res:
    st.stop()

st.markdown("---")
render_section_header("📊", "3. Récapitulatif", "Scores, timeline, replay")

if res.get("mock"):
    st.warning(f"⚠️ Mode mock · model={res.get('model')}")
else:
    st.info(f"Modèle **{res.get('model')}** · device **{res.get('device', '?')}**")

rows = res.get("results") or []
df = pd.DataFrame(rows)
if df.empty:
    st.warning("Aucune détection.")
    st.stop()

k = st.columns(4)
k[0].metric("Détections", len(df))
k[1].metric("Classes", int(df["class"].nunique()) if "class" in df.columns else 0)
k[2].metric("Frames", int(df["frame"].nunique()) if "frame" in df.columns else 0)
k[3].metric(
    "Confiance moy.",
    f"{df['confidence'].mean():.1%}" if "confidence" in df.columns else "—",
)

tab_replay, tab_table, tab_charts = st.tabs(["🎞️ Replay", "📋 Tableau", "📈 Graphiques"])

with tab_replay:
    if st.button("▶️ Rejouer la vidéo annotée", use_container_width=True):
        _replay_with_results(st.session_state.video_path, rows, info, speed=1.0)

    frames = sorted(df["frame"].unique().tolist())
    idx = st.slider("Frame détectée", 0, max(len(frames) - 1, 0), 0)
    fi = int(frames[idx])
    frame = get_frame(st.session_state.video_path, fi) if st.session_state.video_path else None
    dets = df[df["frame"] == fi].to_dict("records")
    if frame is not None:
        st.image(
            draw_detections(frame, dets, show_label=True, show_score=True),
            use_container_width=True,
            caption=f"Frame {fi} · {len(dets)} obj",
        )
    if dets:
        st.dataframe(
            pd.DataFrame(dets)[["class", "confidence", "score_pct", "bbox"]],
            use_container_width=True,
            hide_index=True,
        )

with tab_table:
    show = df.sort_values(["frame", "confidence"], ascending=[True, False])
    cols = [c for c in ["frame", "timestamp", "class", "confidence", "score_pct", "bbox"] if c in show.columns]
    st.dataframe(show[cols], use_container_width=True, hide_index=True, height=400)
    st.download_button("⬇️ CSV", show.to_csv(index=False), "detections.csv", "text/csv")
    st.download_button(
        "⬇️ JSON",
        show.to_json(orient="records", indent=2),
        "detections.json",
        "application/json",
    )

with tab_charts:
    try:
        import plotly.express as px
        fig1 = px.histogram(df, x="class", color="class", title="Par classe")
        fig1.update_layout(template="plotly_dark", showlegend=False, height=320)
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.histogram(df, x="confidence", color="class", nbins=20, title="Scores")
        fig2.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig2, use_container_width=True)
        if "timestamp" in df.columns:
            tl = df.groupby(["timestamp", "class"]).size().reset_index(name="count")
            fig3 = px.bar(tl, x="timestamp", y="count", color="class", title="Timeline")
            fig3.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.caption(str(e))

st.markdown("---")
if st.button("🗑️ Réinitialiser", use_container_width=True):
    for k, v in _defaults.items():
        st.session_state[k] = list(v) if isinstance(v, list) else v
    st.rerun()
