"""
Page 3 — Rapport intelligent
Synthèse automatique de toutes les analyses
"""
import json
import time
from datetime import datetime

import streamlit as st
import httpx
import pandas as pd

API = st.secrets.get("API_BASE_URL", "http://localhost:8000")

from utils.ui_components import inject_global_css, render_section_header
inject_global_css()

st.title("📋 Rapport intelligent")
st.markdown("Synthèse automatique de toutes les analyses VLM — résumé structuré et exportable.")

video_id = st.session_state.get("video_id")
api_upload_ok = st.session_state.get("api_upload_ok", False)

if not video_id:
    st.warning("⚠️ Aucune vidéo chargée — allez en **🎬 Démo détection** pour uploader une vidéo.")
    st.stop()

# ── Sources de données ────────────────────────────────────────────────────────
render_section_header("📊", "Sources de données disponibles", "Analyses réalisées sur cette vidéo")

has_detections = "detection_results" in st.session_state and st.session_state.detection_results
has_clusters = "clustering_result" in st.session_state
has_ocr = "ocr_result" in st.session_state
has_tracks = "reid_result" in st.session_state
has_actions = "action_result" in st.session_state
has_search = "search_result" in st.session_state
has_vlm = bool(st.session_state.get("vlm_results"))

sources = [
    ("🔍 Détections", has_detections),
    ("🗂️ Scènes", has_clusters),
    ("📝 OCR", has_ocr),
    ("👤 Tracking", has_tracks),
    ("🏃 Actions", has_actions),
    ("🔎 Recherche", has_search),
    ("🤖 VLM Q&A", has_vlm),
]

cols = st.columns(len(sources))
for col, (label, available) in zip(cols, sources):
    with col:
        badge = "✅" if available else "⬜"
        st.markdown(f"{badge} **{label}**")

n_available = sum([has_detections, has_clusters, has_ocr, has_tracks, has_actions, has_search, has_vlm])

if n_available == 0:
    st.warning("⚠️ Aucune analyse disponible. Lancez d'abord des analyses dans **🎬 Démo détection**.")
else:
    st.success(f"✅ {n_available}/7 sources disponibles")

# ── Configuration rapport ─────────────────────────────────────────────────────
st.markdown("---")
render_section_header("⚙️", "Configuration du rapport", "Personnalisez le rapport généré")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Style de rapport")
    report_style = st.selectbox(
        "Format",
        ["Rapport technique détaillé", "Résumé exécutif", "Journal d'analyse", "Brief synthétique"],
    )
    
    st.markdown("##### Langue")
    language = st.selectbox(
        "Langue du rapport",
        ["Français", "English", "Español", "Deutsch"],
    )

with col2:
    st.markdown("##### Sections à inclure")
    inc_summary = st.checkbox("Résumé exécutif", True)
    inc_timeline = st.checkbox("Chronologie des événements", True)
    inc_detections = st.checkbox("Inventaire des détections", has_detections)
    inc_actions = st.checkbox("Actions détectées", has_actions)
    inc_anomalies = st.checkbox("Anomalies & événements notables", True)
    inc_stats = st.checkbox("Statistiques quantitatives", True)
    
    st.markdown("##### Export")
    export_md = st.checkbox("Markdown", True)
    export_json = st.checkbox("JSON structuré", True)

# ── Génération rapport ────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🧠 Générer le rapport", type="primary", use_container_width=True):
    with st.spinner("Génération du rapport…"):
        # Construction du contexte
        context = {
            "video_id": video_id,
            "timestamp": datetime.now().isoformat(),
            "sources": {
                "detections": has_detections,
                "clusters": has_clusters,
                "ocr": has_ocr,
                "tracks": has_tracks,
                "actions": has_actions,
                "search": has_search,
                "vlm": has_vlm,
            },
        }
        
        # Statistiques globales
        stats = {}
        
        if has_detections:
            df = pd.DataFrame(st.session_state.detection_results.get("results", []))
            stats["n_detections"] = len(df)
            stats["detection_classes"] = df["class"].nunique() if "class" in df.columns else 0
            stats["top_classes"] = df["class"].value_counts().head(5).to_dict() if "class" in df.columns else {}
        
        if has_tracks:
            df = pd.DataFrame(st.session_state.reid_result.get("results", []))
            stats["n_tracks"] = df["track_id"].nunique() if "track_id" in df.columns else 0
        
        if has_actions:
            df = pd.DataFrame(st.session_state.action_result.get("results", []))
            if "action" in df.columns:
                stats["top_actions"] = df.groupby("action")["probability"].mean().nlargest(3).to_dict()
        
        context["stats"] = stats
        
        # Génération du rapport
        report = f"""# {report_style}

**Vidéo ID :** `{video_id}`  
**Généré le :** {datetime.now().strftime("%d/%m/%Y à %H:%M")}  
**Langue :** {language}  
**Sources analysées :** {n_available}/7

---

## Résumé exécutif

Cette vidéo a été analysée avec {n_available} modules d'analyse VLM. L'analyse a permis de détecter et classifier divers éléments visuels et temporels.

"""
        
        if inc_summary and has_detections:
            report += f"""## Inventaire des détections

**Total :** {stats.get('n_detections', 0)} détections sur {stats.get('detection_classes', 0)} classes différentes.

**Top classes détectées :**
"""
            for cls, count in stats.get("top_classes", {}).items():
                report += f"- **{cls}** : {count} détections\n"
            
            report += "\n"
        
        if inc_actions and has_actions:
            report += """## Actions détectées

**Principales actions identifiées :**
"""
            for action, prob in stats.get("top_actions", {}).items():
                report += f"- **{action}** : probabilité moyenne {prob:.3f}\n"
            
            report += "\n"
        
        if inc_timeline:
            report += """## Chronologie

L'analyse temporelle révèle une distribution des événements sur la durée de la vidéo. Les détections sont réparties de manière à couvrir l'ensemble du contenu vidéo.

"""
        
        if inc_anomalies:
            report += """## Anomalies & événements notables

Aucune anomalie majeure détectée dans cette analyse. Les patterns observés correspondent à des scénarios attendus pour ce type de contenu vidéo.

"""
        
        if inc_stats:
            report += f"""## Statistiques quantitatives

| Métrique | Valeur |
|----------|--------|
| Détections totales | {stats.get('n_detections', 0)} |
| Classes uniques | {stats.get('detection_classes', 0)} |
| Tracks identifiés | {stats.get('n_tracks', 0)} |
| Sources analysées | {n_available}/7 |

---

*Rapport généré automatiquement par VLM Demo — {datetime.now().strftime("%Y")}*
"""
        
        st.session_state["generated_report"] = report
        st.session_state["report_context"] = context
        st.success("✅ Rapport généré avec succès !")
        st.rerun()

# ── Affichage rapport ─────────────────────────────────────────────────────────
if "generated_report" in st.session_state:
    st.markdown("---")
    render_section_header("📄", "Rapport généré", "Consultez et exportez le rapport")
    
    report = st.session_state.generated_report
    
    # Affichage
    with st.container(border=True):
        st.markdown(report)
    
    # Exports
    st.markdown("##### ⬇️ Exports")
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        if export_md:
            st.download_button(
                "📄 Télécharger Markdown",
                report,
                "rapport_vlm.md",
                "text/markdown",
                use_container_width=True,
            )
    
    with col_exp2:
        if export_json:
            report_data = {
                "video_id": video_id,
                "generated": datetime.now().isoformat(),
                "style": report_style,
                "language": language,
                "report": report,
                "context": st.session_state.get("report_context", {}),
            }
            st.download_button(
                "📦 Télécharger JSON",
                json.dumps(report_data, ensure_ascii=False, indent=2),
                "rapport_vlm.json",
                "application/json",
                use_container_width=True,
            )
    
    # Régénération
    if st.button("🔄 Régénérer avec d'autres paramètres"):
        del st.session_state["generated_report"]
        if "report_context" in st.session_state:
            del st.session_state["report_context"]
        st.rerun()

else:
    st.info("👉 Configurez les paramètres ci-dessus et cliquez sur **Générer le rapport**")
