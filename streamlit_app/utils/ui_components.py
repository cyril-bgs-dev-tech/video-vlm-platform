"""
utils/ui_components.py — Composants UX partagés pour toute la plateforme
Cards, badges, polling automatique, video player amélioré, annotations
"""
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass, field

import streamlit as st
import httpx


@dataclass
class Annotation:
    frame_idx: int
    timestamp: float
    label: str
    color: str = "#7F77DD"
    note: str = ""
    created_at: float = field(default_factory=time.time)


def inject_global_css():
    """Injecte le CSS global cohérent pour toute la plateforme."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --primary: #818cf8;
        --primary-light: #a5b4fc;
        --primary-dark: #6366f1;
        --accent: #c084fc;
        --success: #34d399;
        --warning: #fbbf24;
        --error: #f87171;
        --info: #60a5fa;
        --bg-primary: #0a0d14;
        --bg-secondary: #11151f;
        --bg-tertiary: #1a202e;
        --text-primary: #e5e9f0;
        --text-secondary: #8b95a8;
        --border: #222a3a;
        --border-hover: #34405a;
        --shadow: rgba(0, 0, 0, 0.35);
        --radius: 12px;
        --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --mono: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
    }

    html, body, .stApp, [class*="css"] {
        font-family: var(--font);
        letter-spacing: -0.01em;
    }

    code, pre, kbd, .stCode, [data-testid="stCode"] {
        font-family: var(--mono) !important;
    }

    /* Masquer le chrome Streamlit pour un rendu produit */
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        visibility: hidden;
        height: 0;
    }

    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99, 102, 241, 0.13), transparent),
            radial-gradient(ellipse 60% 40% at 90% 0%, rgba(192, 132, 252, 0.07), transparent),
            var(--bg-primary);
    }

    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    h1 {
        background: linear-gradient(90deg, #e5e9f0 30%, var(--primary-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0e16 0%, #10141f 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--primary-light);
        -webkit-text-fill-color: var(--primary-light);
    }

    [data-testid="stSidebarNav"] a,
    [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
        border-radius: 8px;
        transition: background 0.15s ease;
    }

    [data-testid="stSidebarNav"] a:hover,
    [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
        background: rgba(129, 140, 248, 0.1);
    }

    /* Metrics natifs Streamlit */
    [data-testid="stMetric"] {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.9rem 1.1rem;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--border-hover);
        transform: translateY(-1px);
    }

    [data-testid="stMetricValue"] {
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--primary-dark), var(--accent));
        border-radius: 99px;
    }

    /* Images (frames vidéo) */
    [data-testid="stImage"] img {
        border-radius: var(--radius);
        border: 1px solid var(--border);
        box-shadow: 0 8px 32px var(--shadow);
    }

    .status-card {
        border-radius: var(--radius);
        padding: 1rem 1.2rem;
        border: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-size: 0.9rem;
        transition: all 0.25s ease;
        backdrop-filter: blur(10px);
    }

    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px var(--shadow);
    }

    .status-ok {
        background: linear-gradient(135deg, #0d2a1a 0%, #1a3d2e 100%);
        border-color: #1a6b35;
        color: var(--success);
    }

    .status-err {
        background: linear-gradient(135deg, #2a0d0d 0%, #3d1a1a 100%);
        border-color: #8a1a1a;
        color: var(--error);
    }

    .status-warn {
        background: linear-gradient(135deg, #2a2a0d 0%, #3d3d1a 100%);
        border-color: #8a6b1a;
        color: var(--warning);
    }

    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: var(--primary);
        transform: translateY(-3px);
        box-shadow: 0 8px 24px var(--shadow);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-light);
        margin-bottom: 0.3rem;
    }

    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .page-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .page-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }

    .page-card:hover::before {
        transform: scaleX(1);
    }

    .page-card:hover {
        border-color: var(--primary);
        transform: translateY(-4px);
        box-shadow: 0 12px 32px var(--shadow);
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-success {
        background: rgba(74, 222, 128, 0.15);
        color: var(--success);
        border: 1px solid rgba(74, 222, 128, 0.3);
    }

    .badge-warning {
        background: rgba(251, 191, 36, 0.15);
        color: var(--warning);
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    .badge-error {
        background: rgba(248, 113, 113, 0.15);
        color: var(--error);
        border: 1px solid rgba(248, 113, 113, 0.3);
    }

    .badge-info {
        background: rgba(96, 165, 250, 0.15);
        color: var(--info);
        border: 1px solid rgba(96, 165, 250, 0.3);
    }

    .timeline-marker {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        background: var(--bg-tertiary);
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: var(--primary-light);
    }

    .annotation-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
    }

    .progress-container {
        background: var(--bg-tertiary);
        border-radius: 10px;
        padding: 0.3rem;
        margin: 0.5rem 0;
    }

    .progress-bar {
        height: 8px;
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid var(--border);
    }

    .section-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--primary), var(--primary-light));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .section-subtitle {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    .tooltip {
        position: relative;
        cursor: help;
    }

    .tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: var(--bg-tertiary);
        color: var(--text-primary);
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        font-size: 0.75rem;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
        z-index: 1000;
    }

    .tooltip:hover::after {
        opacity: 1;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid var(--border);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: var(--primary);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.18);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-dark) 0%, #8b5cf6 100%);
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35);
        filter: brightness(1.08);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.7rem 1.3rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: var(--bg-secondary);
        border-bottom: 2px solid var(--primary);
    }

    /* Hero (page d'accueil) */
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        background:
            radial-gradient(ellipse 70% 120% at 15% 0%, rgba(99, 102, 241, 0.22), transparent),
            radial-gradient(ellipse 50% 100% at 90% 20%, rgba(192, 132, 252, 0.14), transparent),
            var(--bg-secondary);
        border: 1px solid var(--border);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #f1f5f9 20%, var(--primary-light) 70%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        max-width: 46rem;
    }

    .hero-badges {
        margin-top: 1rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.28rem 0.75rem;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(129, 140, 248, 0.1);
        border: 1px solid rgba(129, 140, 248, 0.25);
        color: var(--primary-light);
    }

    /* Carte évaluation modèle (démo rapide) */
    .model-eval-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
        transition: border-color 0.2s ease;
    }

    .model-eval-card.winner {
        border-color: rgba(52, 211, 153, 0.5);
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.06), var(--bg-secondary) 60%);
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    div[data-testid="stExpander"] summary {
        font-weight: 600;
        color: var(--primary-light);
    }

    .highlight-box {
        background: linear-gradient(135deg, rgba(127, 119, 221, 0.1) 0%, rgba(157, 151, 232, 0.05) 100%);
        border: 1px solid rgba(127, 119, 221, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .info-item {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
    }

    .info-item-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .info-item-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    </style>
    """, unsafe_allow_html=True)


def render_status_card(label: str, ok: bool, detail: str = "", icon: str = "") -> str:
    """Rend une carte de statut avec animation."""
    cls = "status-ok" if ok else "status-err"
    dot = "●" if ok else "○"
    det = f' <span style="opacity:.7">— {detail}</span>' if detail else ""
    icn = f"{icon} " if icon else ""
    return f'<div class="status-card {cls}">{dot} <strong>{icn}{label}</strong>{det}</div>'


def render_metric_card(value: str, label: str, icon: str = "") -> str:
    """Rend une carte de métrique stylisée."""
    icn = f'<div style="font-size:1.5rem;margin-bottom:0.5rem">{icon}</div>' if icon else ""
    return f"""
    <div class="metric-card">
        {icn}
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_badge(text: str, type: str = "info") -> str:
    """Rend un badge coloré (success, warning, error, info)."""
    return f'<span class="badge badge-{type}">{text}</span>'


def render_section_header(icon: str, title: str, subtitle: str = "") -> None:
    """Rend un en-tête de section cohérent."""
    sub = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{icon}</div>
        <div>
            <div class="section-title">{title}</div>
            {sub}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_page_card(num: str, title: str, desc: str, tech: str, has_result: bool = False) -> str:
    """Rend une carte de page pour le dashboard."""
    badge = render_badge("Analyse disponible", "success") if has_result else ""
    return f"""
    <div class="page-card">
        <div style="display:flex;justify-content:space-between;align-items:start">
            <div style="font-size:2rem;color:var(--primary)">{num}</div>
            {badge}
        </div>
        <div style="font-weight:600;font-size:1.1rem;margin:0.8rem 0;color:var(--text-primary)">{title}</div>
        <div style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:0.8rem">{desc}</div>
        <div style="font-size:0.75rem;color:var(--primary-light);padding-top:0.8rem;border-top:1px solid var(--border)">{tech}</div>
    </div>
    """


def poll_task_auto(
    task_id: str,
    api_base: str,
    timeout: float = 300.0,
    interval: float = 1.0,
    label: str = "Traitement en cours",
    on_success: Optional[Callable[[dict], None]] = None,
    on_failure: Optional[Callable[[str], None]] = None,
) -> Optional[dict]:
    """
    Polling automatique avec barre de progression et callbacks.
    Retourne le résultat ou None.
    """
    progress_bar = st.progress(0, text=label)
    status_text = st.empty()
    start_time = time.time()

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time
        progress = min(elapsed / timeout, 0.95)
        progress_bar.progress(progress, text=f"{label} ({int(progress*100)}%)")

        try:
            resp = httpx.get(f"{api_base}/tasks/{task_id}", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "PENDING")
                status_text.caption(f"État : {status}")

                if "progress" in data and data["progress"] is not None:
                    progress_bar.progress(
                        data["progress"] / 100,
                        text=f"{label} ({data['progress']}%)"
                    )

                if status == "SUCCESS":
                    progress_bar.progress(1.0, text="✅ Terminé")
                    status_text.empty()
                    result = data.get("result")
                    if on_success and result:
                        on_success(result)
                    return result
                elif status in ("FAILURE", "REVOKED"):
                    error_msg = data.get("error", status)
                    progress_bar.empty()
                    status_text.error(f"❌ {error_msg}")
                    if on_failure:
                        on_failure(error_msg)
                    return None
        except httpx.ConnectError:
            status_text.warning("API non joignable…")
        except Exception as e:
            status_text.caption(f"Erreur : {e}")

        time.sleep(interval)

    progress_bar.empty()
    status_text.error(f"⏱️ Timeout après {timeout}s")
    return None


def render_annotations_panel(video_id: str, annotations: list[Annotation]) -> list[Annotation]:
    """Rend le panneau d'annotations et permet d'en ajouter."""
    if "annotations" not in st.session_state:
        st.session_state.annotations = []

    render_section_header("📌", "Annotations & Favoris", "Marquez des frames intéressantes")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.annotations:
            for i, ann in enumerate(st.session_state.annotations):
                tc = f"{int(ann.timestamp//60):02d}:{int(ann.timestamp%60):02d}"
                cols = st.columns([0.5, 2, 3, 1])
                with cols[0]:
                    st.markdown(f'<span class="annotation-dot" style="background:{ann.color}"></span>', unsafe_allow_html=True)
                with cols[1]:
                    st.code(tc)
                with cols[2]:
                    st.markdown(f"**{ann.label}** {ann.note}")
                with cols[3]:
                    if st.button("🗑️", key=f"del_ann_{i}"):
                        st.session_state.annotations.pop(i)
                        st.rerun()
        else:
            st.info("Aucune annotation. Ajoutez-en ci-dessous.")

    with col2:
        st.markdown("##### Nouvelle annotation")
        frame_idx = st.number_input("Frame", 0, 100000, 0, key="ann_frame")
        timestamp = st.number_input("Timestamp (s)", 0.0, 10000.0, 0.0, step=0.1, key="ann_time")
        label = st.text_input("Label", "Point d'intérêt", key="ann_label")
        color = st.color_picker("Couleur", "#7F77DD", key="ann_color")
        note = st.text_area("Note", "", key="ann_note")

        if st.button("➕ Ajouter", use_container_width=True):
            new_ann = Annotation(
                frame_idx=frame_idx,
                timestamp=timestamp,
                label=label,
                color=color,
                note=note,
            )
            st.session_state.annotations.append(new_ann)
            st.rerun()

    return st.session_state.annotations


def render_cross_analysis_summary() -> dict:
    """Génère un résumé cross-analyses de toutes les données en session."""
    summary = {
        "has_detections": "detection_result" in st.session_state,
        "has_clusters": "clustering_result" in st.session_state,
        "has_ocr": "ocr_result" in st.session_state,
        "has_tracks": "reid_result" in st.session_state,
        "has_actions": "action_result" in st.session_state,
        "has_search": "search_result" in st.session_state,
        "has_vlm": bool(st.session_state.get("vlm_results")),
    }

    summary["total_analyses"] = sum([
        summary["has_detections"],
        summary["has_clusters"],
        summary["has_ocr"],
        summary["has_tracks"],
        summary["has_actions"],
        summary["has_search"],
        summary["has_vlm"],
    ])

    if summary["has_detections"]:
        import pandas as pd
        df = pd.DataFrame(st.session_state["detection_result"])
        summary["n_detections"] = len(df)
        summary["detection_classes"] = df["class"].nunique() if "class" in df.columns else 0

    if summary["has_tracks"]:
        import pandas as pd
        df = pd.DataFrame(st.session_state["reid_result"])
        summary["n_tracks"] = df["track_id"].nunique() if "track_id" in df.columns else 0

    if summary["has_ocr"]:
        import pandas as pd
        df = pd.DataFrame(st.session_state["ocr_result"])
        summary["n_texts"] = len(df)

    if summary["has_actions"]:
        import pandas as pd
        df = pd.DataFrame(st.session_state["action_result"])
        if "action" in df.columns:
            summary["top_action"] = df.groupby("action")["probability"].mean().idxmax()
        else:
            summary["top_action"] = None
    else:
        summary["top_action"] = None

    return summary


def render_pipeline_launcher(video_id: str, api_base: str) -> None:
    """Lanceur de pipeline unifié pour toutes les analyses."""
    render_section_header("🚀", "Pipeline d'analyse complet", "Lancez toutes les analyses en un clic")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("##### Analyses à lancer")
        run_detection = st.checkbox("🔍 Détection d'objets", True, key="pipe_det")
        run_clustering = st.checkbox("🗂️ Clustering de scènes", True, key="pipe_clust")
        run_ocr = st.checkbox("📝 Extraction OCR", True, key="pipe_ocr")
        run_tracking = st.checkbox("👤 Tracking & Re-ID", True, key="pipe_track")
        run_actions = st.checkbox("🏃 Reconnaissance d'actions", True, key="pipe_act")
        run_search_index = st.checkbox("🔎 Indexation sémantique", True, key="pipe_search")

    with col2:
        st.markdown("##### Paramètres globaux")
        confidence = st.slider("Confiance min", 0.1, 0.9, 0.35, key="pipe_conf")
        fps = st.slider("FPS analyse", 1, 10, 2, key="pipe_fps")

    if st.button("🚀 Lancer le pipeline complet", type="primary", use_container_width=True):
        task_ids = {}

        if run_detection:
            try:
                resp = httpx.post(f"{api_base}/detection/run", json={
                    "video_id": video_id,
                    "model": "grounding_dino_v2",
                    "queries": ["person", "car", "bicycle", "truck"],
                    "confidence": confidence,
                }, timeout=30)
                if resp.status_code == 202:
                    task_ids["detection"] = resp.json()["task_id"]
            except Exception as e:
                st.warning(f"Détection : {e}")

        if run_clustering:
            try:
                resp = httpx.post(f"{api_base}/clustering/scenes", json={
                    "video_id": video_id,
                    "algorithm": "hdbscan",
                }, timeout=30)
                if resp.status_code == 202:
                    task_ids["clustering"] = resp.json()["task_id"]
            except Exception as e:
                st.warning(f"Clustering : {e}")

        if run_ocr:
            try:
                resp = httpx.post(f"{api_base}/analytics/ocr", json={
                    "video_id": video_id,
                    "engine": "paddleocr_v4",
                    "langs": ["fr", "en"],
                }, timeout=30)
                if resp.status_code == 202:
                    task_ids["ocr"] = resp.json()["task_id"]
            except Exception as e:
                st.warning(f"OCR : {e}")

        if run_tracking:
            try:
                resp = httpx.post(f"{api_base}/analytics/reid_track", json={
                    "video_id": video_id,
                    "detector": "yolov9e",
                    "tracker": "botsort",
                }, timeout=30)
                if resp.status_code == 202:
                    task_ids["tracking"] = resp.json()["task_id"]
            except Exception as e:
                st.warning(f"Tracking : {e}")

        if run_actions:
            try:
                resp = httpx.post(f"{api_base}/analytics/actions", json={
                    "video_id": video_id,
                    "model": "videomae_v2_giant",
                }, timeout=30)
                if resp.status_code == 202:
                    task_ids["actions"] = resp.json()["task_id"]
            except Exception as e:
                st.warning(f"Actions : {e}")

        if run_search_index:
            try:
                resp = httpx.post(f"{api_base}/search/index", json={
                    "video_id": video_id,
                    "clip_model": "ViT-L/14",
                    "fps": fps,
                }, timeout=30)
                if resp.status_code == 202:
                    task_ids["search"] = resp.json()["task_id"]
            except Exception as e:
                st.warning(f"Indexation : {e}")

        st.session_state["pipeline_tasks"] = task_ids
        st.success(f"✅ {len(task_ids)} tâches lancées")
        st.rerun()

    if st.session_state.get("pipeline_tasks"):
        st.markdown("##### Progression du pipeline")
        for task_name, task_id in st.session_state["pipeline_tasks"].items():
            with st.expander(f"{task_name} — {task_id[:8]}…"):
                result = poll_task_auto(
                    task_id=task_id,
                    api_base=api_base,
                    timeout=300,
                    label=f"Analyse {task_name}",
                )
                if result:
                    st.success(f"✅ {task_name} terminé")
                    st.json(result)
