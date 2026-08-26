"""
Video VLM Platform — Démonstration de capacités VLM
Workflow : Upload → Sélection signaux → Analyse → Visualisation
"""
import time
import httpx
import streamlit as st

st.set_page_config(
    page_title="VLM Demo — Détection de signaux vidéo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = st.secrets.get("API_BASE_URL", "http://localhost:8000")

# Injection CSS global
from utils.ui_components import inject_global_css
inject_global_css()

# ── Initialisation session state ──────────────────────────────────────────────
_defaults = {
    "video_id": None,
    "video_path": None,
    "video_name": None,
    "api_upload_ok": False,
    "detection_queries": [],
    "analysis_complete": False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 VLM Demo")
    st.markdown("---")
    vid = st.session_state.get("video_id")
    if vid:
        st.success(f"✅ Vidéo chargée")
        st.caption(f"`{vid[:12]}…`")
        if st.session_state.get("api_upload_ok"):
            st.success("API connectée")
        else:
            st.warning("API non connectée")
    else:
        st.info("Aucune vidéo chargée")
    
    st.markdown("---")
    st.markdown("### Navigation")
    st.page_link("pages/1_demo_detection.py", label="🎬 Démo détection")
    st.page_link("pages/2_analyses_detail.py", label="📊 Analyses détaillées")
    st.page_link("pages/3_rapport.py", label="📋 Rapport intelligent")
    st.page_link("pages/4_benchmark_kpis.py", label="📐 Benchmark & KPIs")
    st.page_link("pages/5_compositions_vlm.py", label="🧬 Compositions VLM")
    
    st.markdown("---")
    st.caption("Démonstration VLM — Détection de signaux")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Vision Language Models — Analyse vidéo temps réel</div>
    <div class="hero-subtitle">
        Détectez, suivez et analysez objets, textes, actions et scènes dans vos vidéos.
        Streaming en direct avec bounding boxes, benchmark multi-modèles et rapports automatiques.
    </div>
    <div class="hero-badges">
        <span class="chip">⚡ Live WebSocket</span>
        <span class="chip">🎯 5 modèles de détection</span>
        <span class="chip">🧬 Compositions VLM</span>
        <span class="chip">📐 Benchmark & KPIs</span>
        <span class="chip">🖥️ RTX 5090 · BF16</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Statut services ───────────────────────────────────────────────────────────
st.markdown("### Statut des services")

def _check_service(url: str, timeout: float = 2.0) -> tuple:
    try:
        t0 = time.time()
        r = httpx.get(url, timeout=timeout)
        ms = round((time.time() - t0) * 1000)
        return r.status_code < 500, f"{ms}ms"
    except Exception:
        return False, "—"

@st.cache_data(ttl=10)
def _fetch_health() -> dict:
    try:
        r = httpx.get(f"{API}/health", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

health = _fetch_health()
workers = health.get("workers", {})
n_w = len(workers) if isinstance(workers, dict) else 0
api_ok, api_ms = _check_service(f"{API}/health")

cols = st.columns(3)
with cols[0]:
    status = "status-ok" if api_ok else "status-err"
    dot = "●" if api_ok else "○"
    st.markdown(f'<div class="status-card {status}">{dot} <strong>API FastAPI</strong> — {api_ms}</div>', unsafe_allow_html=True)
with cols[1]:
    status = "status-ok" if n_w > 0 else "status-err"
    dot = "●" if n_w > 0 else "○"
    st.markdown(f'<div class="status-card {status}">{dot} <strong>Workers GPU</strong> — {n_w} actif(s)</div>', unsafe_allow_html=True)
with cols[2]:
    qdrant_ok, qdrant_ms = _check_service("http://qdrant:6333/healthz")
    status = "status-ok" if qdrant_ok else "status-err"
    dot = "●" if qdrant_ok else "○"
    st.markdown(f'<div class="status-card {status}">{dot} <strong>Qdrant</strong> — {qdrant_ms}</div>', unsafe_allow_html=True)

if st.button("🔄 Rafraîchir"):
    st.cache_data.clear()
    st.rerun()

# ── Workflow ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚀 Workflow de démonstration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="highlight-box">
        <div style="font-size:2rem;margin-bottom:0.5rem">📤</div>
        <div style="font-weight:600;font-size:1.1rem;margin-bottom:0.5rem">1. Upload vidéo</div>
        <div style="font-size:0.85rem;color:var(--text-secondary)">
            Chargez une vidéo (MP4, MOV, AVI, MKV, WEBM — max 2 Go)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="highlight-box">
        <div style="font-size:2rem;margin-bottom:0.5rem">🎯</div>
        <div style="font-weight:600;font-size:1.1rem;margin-bottom:0.5rem">2. Sélection signaux</div>
        <div style="font-size:0.85rem;color:var(--text-secondary)">
            Choisissez ce que vous voulez détecter : objets, textes, actions, scènes
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="highlight-box">
        <div style="font-size:2rem;margin-bottom:0.5rem">📊</div>
        <div style="font-weight:600;font-size:1.1rem;margin-bottom:0.5rem">3. Analyse & Résultats</div>
        <div style="font-size:0.85rem;color:var(--text-secondary)">
            Visualisez les détections en temps réel avec overlays et timeline
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Quick Start ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚡ Démarrage rapide")

if not st.session_state.video_id:
    st.info("👉 Commencez par uploader une vidéo dans la page **🎬 Démo détection**")
else:
    st.success(f"✅ Vidéo prête : `{st.session_state.video_id[:12]}…`")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("🎬 Lancer la démo détection", type="primary", use_container_width=True):
            st.switch_page("pages/1_demo_detection.py")
    with col_btn2:
        if st.button("📊 Voir les analyses détaillées", use_container_width=True):
            st.switch_page("pages/2_analyses_detail.py")
    with col_btn3:
        if st.button("📋 Générer un rapport", use_container_width=True):
            st.switch_page("pages/3_rapport.py")

# ── Exemples de détection ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 Exemples de ce que vous pouvez détecter")

examples = [
    ("🚗 Véhicules & Transport", "Voitures, camions, vélos, motos, feux tricolores, panneaux"),
    ("👥 Personnes & Actions", "Personnes, foule, gestes, actions (marche, course, chute)"),
    ("📝 Textes & Enseignes", "Panneaux, enseignes, plaques d'immatriculation, sous-titres"),
    ("🏢 Scènes & Lieux", "Intérieur/extérieur, urbain/rural, types de bâtiments"),
    ("🎯 Objets spécifiques", "Sacs, valises, équipements, outils, animaux"),
]

cols = st.columns(len(examples))
for col, (title, desc) in zip(cols, examples):
    with col:
        st.markdown(f"""
        <div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:10px;padding:1rem;height:100%">
            <div style="font-size:1.5rem;margin-bottom:0.5rem">{title.split()[0]}</div>
            <div style="font-weight:600;margin-bottom:0.3rem">{title.split(' ', 1)[1]}</div>
            <div style="font-size:0.8rem;color:var(--text-secondary)">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:var(--text-secondary);font-size:0.85rem">
    <p>Démonstration VLM — Vision Language Models pour l'analyse vidéo</p>
    <p>API : <a href="http://localhost:8000/docs" style="color:var(--primary-light)">localhost:8000/docs</a></p>
</div>
""", unsafe_allow_html=True)
