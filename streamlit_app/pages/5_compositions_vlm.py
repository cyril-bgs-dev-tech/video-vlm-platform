"""
Page — Compositions VLM (littérature 2025-2026)
Test des 3 compositions conceptuelles pour détection multi-objets:
- Hiérarchique: YOLO-World → LocateAnything → ByteTrack
- Ensembling: GroundingDINO + LocateAnything → fusion NMS
- Itérative: LocateAnything → InternVL3.5 → validation croisée
"""
import json
import uuid
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import httpx

API = st.secrets.get("API_BASE_URL", "http://localhost:8000")

from utils.ui_components import inject_global_css, render_section_header
inject_global_css()

st.title("🧬 Compositions VLM")
st.markdown("Testez les compositions conceptuelles de modèles VLM issues de la littérature 2025-2026.")

# Les trois compositions reposent sur LocateAnything-3B, en cours d'entretien : sans lui, elles
# mêleraient des détections réelles et des détections simulées.
COMPOSITIONS_EN_ENTRETIEN = True
st.warning("🔧 **Compositions en cours d'entretien.** Les trois compositions reposent sur LocateAnything-3B, "
           "son code distant est incompatible avec transformers v5 : le modèle charge et génère, mais sa sortie est dégénérée et ne produit aucune boîte. Elles restent décrites ici, mais leur lancement est suspendu jusqu'à sa réparation.")

# ── Informations sur les compositions ─────────────────────────────────────────
with st.expander("📚 Compositions disponibles (littérature 2025-2026)", expanded=False):
    st.markdown("""
    ### 1. 🏗️ Hiérarchique (2COOOL ICCV 2025 — 2ème place)
    **Architecture:** YOLO-World (rapide 15 FPS) → LocateAnything-3B (sémantique 3 FPS) → ByteTrack (tracking)
    
    **Avantage:** 60-80% réduction charge computationnelle vs VLM seul
    
    **Cas d'usage:** Flux vidéo temps réel avec compréhension sémantique
    
    ---
    
    ### 2. 🗳️ Ensembling (Multiple papiers 2025-2026)
    **Architecture:** GroundingDINO + LocateAnything-3B → fusion NMS + voting
    
    **Avantage:** Robustesse accrue, réduction faux positifs
    
    **Cas d'usage:** Détection critique où la précision est primordiale
    
    ---
    
    ### 3. 🔄 Itérative (LongShOTAgent, ScreenSeekeR 2025-2026)
    **Architecture:** LocateAnything-3B → InternVL3.5-8B (raffinement) → validation croisée
    
    **Avantage:** Haute précision pour scènes complexes
    
    **Cas d'usage:** Analyse détaillée de scènes denses ou ambiguës
    """)

# ── Initialisation session state ──────────────────────────────────────────────
_defaults = {
    "video_id": None,
    "video_path": None,
    "video_name": None,
    "api_upload_ok": False,
    "composition_queries": [],
    "composition_results": None,
    "analysis_running": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Section 1 : Upload ────────────────────────────────────────────────────────
render_section_header("📤", "Étape 1 : Charger une vidéo", "Glissez-déposez ou parcourez vos fichiers")

uploaded = st.file_uploader(
    label="Vidéo (MP4, MOV, AVI, MKV, WEBM — max 2 Go)",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    accept_multiple_files=False,
)

if uploaded is not None and st.session_state.video_path is None:
    suffix = Path(uploaded.name).suffix or ".mp4"
    local_id = str(uuid.uuid4())

    with st.spinner("💾 Enregistrement local…"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getbuffer())
        tmp.close()

    st.session_state.video_path = tmp.name
    st.session_state.video_id = local_id
    st.session_state.video_name = uploaded.name
    st.session_state.api_upload_ok = False

    with st.spinner("⬆️ Transmission vers l'API…"):
        try:
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    f"{API}/upload",
                    files={"file": (uploaded.name, open(tmp.name, "rb"), uploaded.type or "video/mp4")},
                )
            if resp.status_code == 200:
                api_id = resp.json().get("video_id", local_id)
                st.session_state.video_id = api_id
                st.session_state.api_upload_ok = True
                st.success(f"✅ Vidéo transmise — ID : `{api_id[:12]}…`")
            else:
                st.warning(f"⚠️ API a retourné HTTP {resp.status_code}. Mode local activé.")
        except Exception as e:
            st.warning(f"⚠️ API non joignable : {e}. Mode local activé.")

    st.rerun()

# Retry upload si échec
if st.session_state.video_path and not st.session_state.api_upload_ok:
    col_retry, col_info = st.columns([1, 3])
    with col_retry:
        if st.button("🔄 Réessayer l'upload API", use_container_width=True):
            try:
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        f"{API}/upload",
                        files={"file": (st.session_state.video_name, open(st.session_state.video_path, "rb"), "video/mp4")},
                    )
                if resp.status_code == 200:
                    api_id = resp.json().get("video_id", st.session_state.video_id)
                    st.session_state.video_id = api_id
                    st.session_state.api_upload_ok = True
                    st.success(f"✅ Upload réussi — ID : `{api_id[:12]}…`")
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur : {e}")
    with col_info:
        st.caption("L'API n'a pas reçu la vidéo. Les analyses GPU nécessitent l'API.")

# ── Section 2 : Sélection composition ─────────────────────────────────────────
if st.session_state.video_id:
    st.markdown("---")
    render_section_header("🧬", "Étape 2 : Choisir une composition", "Sélectionnez la stratégie de composition")

    composition_options = {
        "🏗️ Hiérarchique (recommandé)": {
            "value": "hierarchical",
            "description": "YOLO-World → LocateAnything → ByteTrack",
            "details": "60-80% réduction charge — Idéal flux vidéo temps réel",
            "models": ["YOLO-World v2", "LocateAnything-3B", "ByteTrack"],
            "vram": "~16GB",
        },
        "🗳️ Ensembling": {
            "value": "ensembling",
            "description": "GroundingDINO + LocateAnything → fusion NMS",
            "details": "Robustesse accrue — Réduction faux positifs",
            "models": ["GroundingDINO", "LocateAnything-3B"],
            "vram": "~14GB",
        },
        "🔄 Itérative": {
            "value": "iterative_refinement",
            "description": "LocateAnything → InternVL3.5 → validation",
            "details": "Haute précision — Scènes complexes",
            "models": ["LocateAnything-3B", "InternVL3.5-8B"],
            "vram": "~28GB",
        },
    }

    comp_names = list(composition_options.keys())
    selected_comp_name = st.selectbox(
        "Composition",
        comp_names,
        index=0,
        help="Choisissez la stratégie de composition selon vos besoins"
    )

    selected_comp = composition_options[selected_comp_name]
    
    col_desc, col_models, col_vram = st.columns([2, 2, 1])
    with col_desc:
        st.markdown(f"**{selected_comp['description']}**")
        st.caption(selected_comp['details'])
    with col_models:
        st.markdown("**Modèles:**")
        for model in selected_comp['models']:
            st.caption(f"• {model}")
    with col_vram:
        st.metric("VRAM", selected_comp['vram'])

    # ── Section 3 : Requêtes ──────────────────────────────────────────────────
    st.markdown("---")
    render_section_header("🎯", "Étape 3 : Définir les requêtes", "Ce que vous voulez détecter")

    st.markdown("##### Presets rapides")
    preset_cols = st.columns(4)
    
    presets = {
        "🚗 Véhicules": ["car", "truck", "bicycle", "motorcycle", "bus"],
        "👥 Personnes": ["person", "crowd", "face"],
        "📝 Textes": ["sign", "text", "label", "plate"],
        "🎯 Personnalisé": None,
    }

    for col, (preset_name, preset_queries) in zip(preset_cols, presets.items()):
        with col:
            if st.button(preset_name, use_container_width=True, key=f"preset_{preset_name}"):
                if preset_queries:
                    st.session_state.composition_queries = preset_queries
                    st.rerun()

    custom_queries = st.text_area(
        "Éléments à détecter (un par ligne)",
        value="\n".join(st.session_state.composition_queries) if st.session_state.composition_queries else "person\ncar\nbicycle",
        height=150,
    )
    
    if st.button("✅ Appliquer les requêtes"):
        st.session_state.composition_queries = [q.strip() for q in custom_queries.splitlines() if q.strip()]
        st.success(f"✅ {len(st.session_state.composition_queries)} requêtes définies")
        st.rerun()

    if st.session_state.composition_queries:
        st.markdown("**Requêtes actives :**")
        query_badges = " ".join([f'<span class="badge badge-info">{q}</span>' for q in st.session_state.composition_queries])
        st.markdown(query_badges, unsafe_allow_html=True)

    # ── Section 4 : Paramètres ────────────────────────────────────────────────
    st.markdown("---")
    render_section_header("⚙️", "Étape 4 : Paramètres", "Ajustez les paramètres de la composition")

    col1, col2, col3 = st.columns(3)
    with col1:
        confidence = st.slider("Confiance minimale", 0.1, 0.9, 0.35, 0.05)
    with col2:
        enable_tracking = st.checkbox("Tracking inter-frames", True)
    with col3:
        if selected_comp["value"] == "hierarchical":
            fast_fps = st.slider("FPS rapide (YOLO)", 5, 30, 15)
            semantic_fps = st.slider("FPS sémantique (VLM)", 1, 10, 3)
        elif selected_comp["value"] == "ensembling":
            iou_threshold = st.slider("Seuil IoU fusion", 0.3, 0.9, 0.5, 0.05)
        elif selected_comp["value"] == "iterative_refinement":
            refinement_passes = st.slider("Passes de raffinement", 1, 5, 2)

    # ── Section 5 : Lancement ─────────────────────────────────────────────────
    st.markdown("---")
    render_section_header("🚀", "Étape 5 : Lancer la composition", "Exécuter la composition sur la vidéo")

    launch_disabled = (COMPOSITIONS_EN_ENTRETIEN or not st.session_state.api_upload_ok
                       or not st.session_state.composition_queries)
    
    if st.button(
        f"🧬 Lancer la composition {selected_comp_name}",
        type="primary",
        use_container_width=True,
        disabled=launch_disabled,
    ):
        st.session_state.analysis_running = True
        
        with st.spinner(f"Composition {selected_comp['value']} en cours…"):
            try:
                payload = {
                    "video_id": st.session_state.video_id,
                    "composition_type": selected_comp["value"],
                    "queries": st.session_state.composition_queries,
                    "confidence": confidence,
                    "track": enable_tracking,
                }
                
                if selected_comp["value"] == "hierarchical":
                    payload["fast_fps"] = fast_fps
                    payload["semantic_fps"] = semantic_fps
                elif selected_comp["value"] == "ensembling":
                    payload["iou_threshold"] = iou_threshold
                elif selected_comp["value"] == "iterative_refinement":
                    payload["refinement_passes"] = refinement_passes
                
                resp = httpx.post(
                    f"{API}/composition/run",
                    json=payload,
                    timeout=600,
                )
                
                if resp.status_code == 202:
                    task_id = resp.json()["task_id"]
                    st.info(f"⏳ Tâche lancée : `{task_id[:12]}…`")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    start_time = time.time()
                    timeout = 600
                    
                    while time.time() - start_time < timeout:
                        elapsed = time.time() - start_time
                        progress = min(elapsed / timeout, 0.95)
                        progress_bar.progress(progress, text=f"Analyse en cours ({int(progress*100)}%)")
                        
                        try:
                            task_resp = httpx.get(f"{API}/tasks/{task_id}", timeout=10)
                            if task_resp.status_code == 200:
                                task_data = task_resp.json()
                                status = task_data.get("status", "PENDING")
                                status_text.caption(f"État : {status}")
                                
                                if status == "SUCCESS":
                                    progress_bar.progress(1.0, text="✅ Analyse terminée")
                                    st.session_state.composition_results = task_data.get("result")
                                    st.session_state.analysis_running = False
                                    st.success("✅ Composition terminée !")
                                    st.rerun()
                                elif status == "FAILURE":
                                    st.error(f"❌ Erreur : {task_data.get('error')}")
                                    st.session_state.analysis_running = False
                                    break
                        except Exception as e:
                            status_text.warning(f"Polling error: {e}")
                        
                        time.sleep(1.0)
                    
                    progress_bar.empty()
                    status_text.empty()
                else:
                    st.error(f"Erreur API : {resp.text}")
                    st.session_state.analysis_running = False
                    
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.session_state.analysis_running = False

    if launch_disabled:
        if not st.session_state.api_upload_ok:
            st.caption("🔒 Nécessite l'API connectée")
        elif not st.session_state.composition_queries:
            st.caption("🔒 Définissez d'abord les requêtes")

    # ── Section 6 : Résultats ─────────────────────────────────────────────────
    if st.session_state.composition_results:
        st.markdown("---")
        render_section_header("📊", "Étape 6 : Résultats", "Visualisation des détections")

        results = st.session_state.composition_results
        
        if "error" in results:
            st.error(f"Erreur : {results['error']}")
        else:
            detections = results.get("results", [])
            composition_type = results.get("composition", "unknown")
            models_used = results.get("models", [])
            
            st.info(f"**Composition:** {composition_type} | **Modèles:** {', '.join(models_used)}")
            
            if not detections:
                st.warning("Aucune détection trouvée.")
            else:
                import pandas as pd
                df = pd.DataFrame(detections)
                
                # Métriques
                st.markdown("##### Métriques")
                metric_cols = st.columns(5)
                
                with metric_cols[0]:
                    st.metric("Total détections", len(df))
                with metric_cols[1]:
                    st.metric("Classes uniques", df["class"].nunique() if "class" in df.columns else 0)
                with metric_cols[2]:
                    st.metric("Frames analysées", df["frame"].nunique() if "frame" in df.columns else 0)
                with metric_cols[3]:
                    avg_conf = df["confidence"].mean() if "confidence" in df.columns else 0
                    st.metric("Confiance moyenne", f"{avg_conf:.2f}")
                with metric_cols[4]:
                    if "votes" in df.columns:
                        avg_votes = df["votes"].mean()
                        st.metric("Votes moyens (ensemble)", f"{avg_votes:.1f}")
                    elif "cross_validated" in df.columns:
                        validated_pct = df["cross_validated"].mean() * 100
                        st.metric("Validation croisée", f"{validated_pct:.0f}%")
                    else:
                        n_tracks = df["track_id"].nunique() if "track_id" in df.columns else 0
                        st.metric("Tracks uniques", n_tracks)
                
                # Timeline
                st.markdown("##### Timeline des détections")
                
                if "timestamp" in df.columns and "class" in df.columns:
                    tab1, tab2, tab3 = st.tabs(["📈 Vue temporelle", "📋 Tableau", "🎥 Player"])
                    
                    with tab1:
                        import plotly.express as px
                        timeline_data = df.groupby(["timestamp", "class"]).size().reset_index(name="count")
                        fig = px.bar(
                            timeline_data,
                            x="timestamp",
                            y="count",
                            color="class",
                            title="Détections par moment",
                            labels={"timestamp": "Temps (s)", "count": "Nombre"},
                        )
                        fig.update_layout(height=400, template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        cols_to_show = ["frame", "timestamp", "class", "confidence"]
                        if "source" in df.columns:
                            cols_to_show.append("source")
                        if "votes" in df.columns:
                            cols_to_show.append("votes")
                        if "cross_validated" in df.columns:
                            cols_to_show.append("cross_validated")
                        
                        st.dataframe(df[cols_to_show].head(100), use_container_width=True)
                        
                        col_exp1, col_exp2 = st.columns(2)
                        with col_exp1:
                            st.download_button(
                                "⬇️ Export CSV",
                                df.to_csv(index=False),
                                "composition_results.csv",
                                "text/csv",
                                use_container_width=True,
                            )
                        with col_exp2:
                            st.download_button(
                                "⬇️ Export JSON",
                                df.to_json(orient="records", indent=2),
                                "composition_results.json",
                                "application/json",
                                use_container_width=True,
                            )
                    
                    with tab3:
                        if st.session_state.video_path:
                            cap = cv2.VideoCapture(st.session_state.video_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            cap.release()
                            
                            frame_idx = st.slider("Frame", 0, total_frames - 1, 0)
                            
                            cap = cv2.VideoCapture(st.session_state.video_path)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                            cap.release()
                            
                            if ret:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_detections = df[df["frame"] == frame_idx] if "frame" in df.columns else pd.DataFrame()
                                
                                if not frame_detections.empty:
                                    for _, det in frame_detections.iterrows():
                                        bbox = det.get("bbox", [0, 0, 0, 0])
                                        if len(bbox) == 4:
                                            x1, y1, x2, y2 = [int(v) for v in bbox]
                                            label = det.get("class", "obj")
                                            conf = det.get("confidence", 0)
                                            source = det.get("source", "")
                                            
                                            color = (0, 255, 0) if source == "yolo_world" else (255, 0, 0) if source == "locate_anything" else (0, 255, 255)
                                            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                                            
                                            text = f"{label} {conf:.2f}"
                                            if source:
                                                text += f" ({source})"
                                            cv2.putText(frame_rgb, text, (x1, max(y1 - 10, 0)),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                st.image(frame_rgb, use_container_width=True)
                                timestamp = frame_idx / fps
                                st.caption(f"Frame {frame_idx} — {timestamp:.2f}s — {len(frame_detections)} détection(s)")

    # Reset
    st.markdown("---")
    if st.button("🗑️ Réinitialiser", use_container_width=True):
        for k in _defaults:
            st.session_state[k] = _defaults[k]
        st.rerun()

else:
    st.info("👉 Uploadez une vidéo pour commencer")
