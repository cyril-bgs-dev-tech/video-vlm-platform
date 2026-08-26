"""
Page 2 — Analyses détaillées
Visualisations avancées par type d'analyse
"""
import streamlit as st
import httpx
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

API = st.secrets.get("API_BASE_URL", "http://localhost:8000")

from utils.ui_components import inject_global_css, render_section_header
inject_global_css()

st.title("📊 Analyses détaillées")
st.markdown("Visualisations avancées des détections VLM — explorez chaque type d'analyse.")

video_id = st.session_state.get("video_id")
api_upload_ok = st.session_state.get("api_upload_ok", False)

if not video_id:
    st.warning("⚠️ Aucune vidéo chargée — allez en **🎬 Démo détection** pour uploader une vidéo.")
    st.stop()

# ── Tabs d'analyses ───────────────────────────────────────────────────────────
tab_det, tab_track, tab_scene, tab_action, tab_search = st.tabs([
    "🔍 Détections",
    "👤 Tracking",
    "🗂️ Scènes",
    "🏃 Actions",
    "🔎 Recherche",
])

# ── Tab 1 : Détections ────────────────────────────────────────────────────────
with tab_det:
    render_section_header("🔍", "Détections d'objets", "GroundingDINO · YOLO-World")
    
    detection_result = st.session_state.get("detection_results")
    
    if detection_result and "results" in detection_result:
        df = pd.DataFrame(detection_result["results"])
        
        if not df.empty:
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total détections", len(df))
            with col2:
                st.metric("Classes uniques", df["class"].nunique())
            with col3:
                st.metric("Confiance moyenne", f"{df['confidence'].mean():.3f}")
            with col4:
                st.metric("Frames couvertes", df["frame"].nunique())
            
            # Distribution par classe
            st.markdown("##### Distribution par classe")
            class_dist = df["class"].value_counts().reset_index()
            class_dist.columns = ["Classe", "Count"]
            
            fig = px.bar(
                class_dist, x="Classe", y="Count",
                color="Count", color_continuous_scale="Viridis",
                template="plotly_dark",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            # Timeline
            st.markdown("##### Timeline des détections")
            if "timestamp" in df.columns:
                df_sorted = df.sort_values("timestamp")
                
                fig2 = go.Figure()
                for cls in df["class"].unique()[:5]:  # Top 5 classes
                    cls_data = df[df["class"] == cls]
                    fig2.add_trace(go.Scatter(
                        x=cls_data["timestamp"],
                        y=[cls] * len(cls_data),
                        mode="markers",
                        name=cls,
                        marker=dict(size=cls_data["confidence"] * 15, opacity=0.7),
                    ))
                
                fig2.update_layout(
                    height=300,
                    template="plotly_dark",
                    xaxis_title="Temps (s)",
                    showlegend=True,
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Tableau filtrable
            st.markdown("##### Données détaillées")
            selected_classes = st.multiselect(
                "Filtrer par classe",
                df["class"].unique(),
                default=df["class"].unique()[:3],
            )
            
            df_filtered = df[df["class"].isin(selected_classes)]
            st.dataframe(
                df_filtered[["frame", "timestamp", "class", "confidence", "bbox"]].head(200),
                use_container_width=True,
            )
        else:
            st.info("Aucune détection disponible. Lancez une analyse dans la page **🎬 Démo détection**.")
    else:
        st.info("Aucune analyse de détection disponible. Lancez une analyse dans la page **🎬 Démo détection**.")

# ── Tab 2 : Tracking ──────────────────────────────────────────────────────────
with tab_track:
    render_section_header("👤", "Tracking & Re-ID", "Suivi multi-objets dans le temps")
    
    reid_result = st.session_state.get("reid_result")
    
    if reid_result and "results" in reid_result:
        df = pd.DataFrame(reid_result["results"])
        
        if not df.empty and "track_id" in df.columns:
            st.metric("Tracks uniques", df["track_id"].nunique())
            
            # Trajectoires
            st.markdown("##### Trajectoires des tracks")
            fig = px.scatter(
                df, x="x_center", y="y_center",
                color=df["track_id"].astype(str),
                animation_frame="frame",
                template="plotly_dark",
                title="Positions des tracks dans le temps",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Durée des tracks
            st.markdown("##### Durée des tracks")
            track_durations = df.groupby("track_id").agg(
                start_frame=("frame", "min"),
                end_frame=("frame", "max"),
            ).reset_index()
            track_durations["duration_frames"] = track_durations["end_frame"] - track_durations["start_frame"]
            
            fig2 = px.bar(
                track_durations, x="track_id", y="duration_frames",
                template="plotly_dark",
                title="Durée de chaque track (en frames)",
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Aucune donnée de tracking. Lancez une analyse de tracking dans la page **🎬 Démo détection**.")
    else:
        st.info("Aucune analyse de tracking disponible. Lancez une analyse dans la page **🎬 Démo détection**.")

# ── Tab 3 : Scènes ────────────────────────────────────────────────────────────
with tab_scene:
    render_section_header("🗂️", "Clustering de scènes", "Découpe sémantique automatique")
    
    clustering_result = st.session_state.get("clustering_result")
    
    if clustering_result and "results" in clustering_result:
        df = pd.DataFrame(clustering_result["results"])
        
        if not df.empty:
            st.metric("Scènes détectées", df["cluster"].nunique())
            
            # UMAP 2D
            if "umap_x" in df.columns and "umap_y" in df.columns:
                st.markdown("##### Projection UMAP des scènes")
                fig = px.scatter(
                    df, x="umap_x", y="umap_y",
                    color="cluster",
                    hover_data=["timestamp", "frame_idx"],
                    template="plotly_dark",
                    title="Projection 2D des embeddings CLIP",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Timeline des scènes
            st.markdown("##### Timeline des scènes")
            if "timestamp" in df.columns:
                fig2 = px.timeline(
                    df, x_start="timestamp", x_end="timestamp",
                    y="cluster",
                    color="cluster",
                    template="plotly_dark",
                    title="Découpe temporelle des scènes",
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Aucune donnée de clustering.")
    else:
        st.info("Aucune analyse de clustering disponible.")

# ── Tab 4 : Actions ───────────────────────────────────────────────────────────
with tab_action:
    render_section_header("🏃", "Reconnaissance d'actions", "Classification temporelle")
    
    action_result = st.session_state.get("action_result")
    
    if action_result and "results" in action_result:
        df = pd.DataFrame(action_result["results"])
        
        if not df.empty and "action" in df.columns:
            st.metric("Actions détectées", df["action"].nunique())
            
            # Timeline des actions
            st.markdown("##### Timeline des actions")
            if "timestamp" in df.columns:
                pivot = df.pivot_table(
                    index="timestamp", columns="action",
                    values="probability", aggfunc="max"
                ).fillna(0)
                
                fig = go.Figure()
                for action in pivot.columns[:5]:  # Top 5 actions
                    fig.add_trace(go.Scatter(
                        x=pivot.index, y=pivot[action],
                        mode="lines", name=action,
                    ))
                
                fig.update_layout(
                    height=350,
                    template="plotly_dark",
                    xaxis_title="Temps (s)",
                    yaxis_title="Probabilité",
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Ranking
            st.markdown("##### Ranking des actions")
            action_ranking = df.groupby("action")["probability"].mean().sort_values(ascending=False)
            
            fig2 = px.bar(
                x=action_ranking.values, y=action_ranking.index,
                orientation="h",
                template="plotly_dark",
                title="Score moyen par action",
            )
            fig2.update_layout(height=300, xaxis_title="Probabilité moyenne")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Aucune donnée d'actions.")
    else:
        st.info("Aucune analyse d'actions disponible.")

# ── Tab 5 : Recherche ─────────────────────────────────────────────────────────
with tab_search:
    render_section_header("🔎", "Recherche sémantique", "Recherche par texte ou image")
    
    search_result = st.session_state.get("search_result")
    
    if search_result:
        df = pd.DataFrame(search_result)
        
        if not df.empty:
            st.metric("Résultats trouvés", len(df))
            
            # Timeline des résultats
            st.markdown("##### Résultats dans la timeline")
            if "timestamp" in df.columns and "similarity" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["similarity"],
                    mode="markers+lines",
                    marker=dict(
                        size=df["similarity"] * 20,
                        color=df["similarity"],
                        colorscale="Viridis",
                        showscale=True,
                    ),
                ))
                
                fig.update_layout(
                    height=300,
                    template="plotly_dark",
                    xaxis_title="Temps (s)",
                    yaxis_title="Similarité",
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Tableau
            st.markdown("##### Résultats détaillés")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Aucun résultat de recherche.")
    else:
        st.info("Aucune recherche effectuée. Utilisez la page **🎬 Démo détection** pour indexer et rechercher.")
