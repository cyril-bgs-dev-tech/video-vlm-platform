"""
Page 4 — Benchmark & KPIs DataOps
Monitoring complet pour modèles de détection d'objets en flux vidéo.
"""
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Benchmark & KPIs — VLM Platform", page_icon="📐", layout="wide")

from utils.ui_components import inject_global_css, render_section_header
inject_global_css()

# ── Données de référence (basées sur benchmarks publics + profils RTX 5090) ──

# Modèles en cours d'entretien : grisés et retirés du comparatif tant que leur détection ne fonctionne pas.
EN_ENTRETIEN = {
    "LocateAnything-3B": "En cours d'entretien : son code distant est incompatible avec transformers v5 : le modèle charge et génère, mais sa sortie est dégénérée et ne produit aucune boîte.",
}

BASE_MODELS = {
    "LocateAnything-3B": {
        "params": "3B",
        "vram_gb": 12,
        "fps_1080p": 18.5,
        "fps_720p": 28.0,
        "latency_ms": 54,
        "map50": 0.572,
        "map50_95": 0.381,
        "precision": 0.612,
        "recall": 0.548,
        "mota": 0.498,
        "id_switches_per_1k": 12,
        "power_w": 320,
        "model_size_mb": 6200,
        "category": "VLM Grounding",
    },
    "Mini-InternVL-4B": {
        "params": "4B",
        "vram_gb": 9,
        "fps_1080p": 14.2,
        "fps_720p": 22.5,
        "latency_ms": 70,
        "map50": 0.558,
        "map50_95": 0.365,
        "precision": 0.598,
        "recall": 0.531,
        "mota": 0.479,
        "id_switches_per_1k": 15,
        "power_w": 295,
        "model_size_mb": 8100,
        "category": "VLM Grounding",
    },
    "InternVL3.5-8B": {
        "params": "8B",
        "vram_gb": 15,
        "fps_1080p": 8.3,
        "fps_720p": 13.1,
        "latency_ms": 120,
        "map50": 0.621,
        "map50_95": 0.425,
        "precision": 0.654,
        "recall": 0.589,
        "mota": 0.541,
        "id_switches_per_1k": 8,
        "power_w": 380,
        "model_size_mb": 16200,
        "category": "VLM Grounding",
    },
    "GroundingDINO v2": {
        "params": "0.5B",
        "vram_gb": 2,
        "fps_1080p": 32.0,
        "fps_720p": 48.0,
        "latency_ms": 31,
        "map50": 0.487,
        "map50_95": 0.312,
        "precision": 0.534,
        "recall": 0.461,
        "mota": 0.412,
        "id_switches_per_1k": 22,
        "power_w": 210,
        "model_size_mb": 1050,
        "category": "Classical",
    },
    "YOLO-World v2-X": {
        "params": "0.1B",
        "vram_gb": 4,
        "fps_1080p": 55.0,
        "fps_720p": 85.0,
        "latency_ms": 18,
        "map50": 0.523,
        "map50_95": 0.348,
        "precision": 0.567,
        "recall": 0.498,
        "mota": 0.456,
        "id_switches_per_1k": 18,
        "power_w": 195,
        "model_size_mb": 240,
        "category": "Classical",
    },
}

# ── Datasets d'évaluation — 20 domaines ──────────────────────────────────────
# difficulty : multiplicateur global de précision (petits objets, occlusions…)
# speed_factor : multiplicateur de débit (résolution / complexité des images)
# affinity : bonus/malus par catégorie de modèle sur ce domaine
# weights : pondération du score de sélection automatique (somme = 1.0)
DATASETS = {
    "COCO val2017": {
        "icon": "🧰", "domain": "Généraliste",
        "desc": "80 classes du quotidien — référence standard de la détection.",
        "difficulty": 1.00, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 1.00, "Classical": 1.00},
        "weights": {"acc": 0.35, "speed": 0.30, "track": 0.20, "res": 0.15},
        "traits": ["scènes variées", "objets moyens/grands", "annotations denses"],
        "reco": "Profil équilibré : le meilleur compromis précision/vitesse l'emporte.",
    },
    "MOT20": {
        "icon": "🚶", "domain": "Foule urbaine / tracking",
        "desc": "Piétons très denses (>100 personnes/frame), suivi multi-objets.",
        "difficulty": 0.82, "speed_factor": 0.95,
        "affinity": {"VLM Grounding": 1.02, "Classical": 0.95},
        "weights": {"acc": 0.25, "speed": 0.20, "track": 0.45, "res": 0.10},
        "traits": ["occlusions massives", "densité extrême", "MOTA critique"],
        "reco": "Le tracking domine : privilégier MOTA et stabilité des IDs.",
    },
    "VisDrone": {
        "icon": "🛸", "domain": "Drone / vue aérienne",
        "desc": "Vues aériennes obliques, objets minuscules, forte densité urbaine.",
        "difficulty": 0.72, "speed_factor": 0.90,
        "affinity": {"VLM Grounding": 1.08, "Classical": 0.88},
        "weights": {"acc": 0.45, "speed": 0.25, "track": 0.20, "res": 0.10},
        "traits": ["objets < 20 px", "angles inhabituels", "fond chargé"],
        "reco": "Petits objets : les VLM grounding gardent le contexte spatial.",
    },
    "CityPersons": {
        "icon": "🏙️", "domain": "Piétons urbains",
        "desc": "Piétons en environnement urbain européen, occlusions partielles.",
        "difficulty": 0.90, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 1.00, "Classical": 1.00},
        "weights": {"acc": 0.35, "speed": 0.30, "track": 0.25, "res": 0.10},
        "traits": ["classe unique", "occlusions", "échelles variées"],
        "reco": "Mono-classe : la vitesse redevient un facteur discriminant.",
    },
    "BDD100K": {
        "icon": "🚗", "domain": "Conduite autonome",
        "desc": "Caméra embarquée : véhicules, piétons, feux — jour/nuit/pluie.",
        "difficulty": 0.88, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 1.00, "Classical": 1.02},
        "weights": {"acc": 0.30, "speed": 0.40, "track": 0.20, "res": 0.10},
        "traits": ["temps réel obligatoire", "météo variable", "10 classes"],
        "reco": "Contrainte temps réel forte : ≥25 FPS est éliminatoire.",
    },
    "CrowdHuman": {
        "icon": "👥", "domain": "Foule dense",
        "desc": "~23 personnes/image, chevauchements extrêmes, NMS difficile.",
        "difficulty": 0.78, "speed_factor": 0.95,
        "affinity": {"VLM Grounding": 1.05, "Classical": 0.92},
        "weights": {"acc": 0.45, "speed": 0.20, "track": 0.25, "res": 0.10},
        "traits": ["boxes chevauchantes", "full-body + head", "recall critique"],
        "reco": "Chevauchements : le raisonnement contextuel VLM réduit les fusions de boxes.",
    },
    "WIDER FACE": {
        "icon": "🙂", "domain": "Visages",
        "desc": "Visages en foule, échelles 4 px → 1000 px, poses extrêmes.",
        "difficulty": 0.75, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 1.06, "Classical": 0.90},
        "weights": {"acc": 0.50, "speed": 0.25, "track": 0.15, "res": 0.10},
        "traits": ["très petits visages", "flou/occlusion", "expression/pose"],
        "reco": "Précision d'abord : les micro-visages pénalisent les détecteurs classiques.",
    },
    "DOTA v2.0": {
        "icon": "🛰️", "domain": "Imagerie satellite",
        "desc": "Images satellite 4000×4000, objets orientés (avions, navires, ponts).",
        "difficulty": 0.68, "speed_factor": 0.80,
        "affinity": {"VLM Grounding": 1.10, "Classical": 0.85},
        "weights": {"acc": 0.55, "speed": 0.15, "track": 0.10, "res": 0.20},
        "traits": ["boxes orientées", "très haute résolution", "18 classes"],
        "reco": "Analyse offline haute précision : la latence compte peu, la VRAM oui (tuiles).",
    },
    "ExDark": {
        "icon": "🌙", "domain": "Vision nocturne",
        "desc": "12 classes en très faible luminosité — bruit capteur élevé.",
        "difficulty": 0.70, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 1.08, "Classical": 0.90},
        "weights": {"acc": 0.45, "speed": 0.30, "track": 0.15, "res": 0.10},
        "traits": ["faible contraste", "bruit ISO", "contexte sémantique utile"],
        "reco": "Faible lumière : le prior sémantique des VLM compense le signal dégradé.",
    },
    "FLIR ADAS": {
        "icon": "🔥", "domain": "Imagerie thermique",
        "desc": "Caméra thermique embarquée : personnes, véhicules, vélos.",
        "difficulty": 0.74, "speed_factor": 1.10,
        "affinity": {"VLM Grounding": 0.95, "Classical": 1.02},
        "weights": {"acc": 0.35, "speed": 0.35, "track": 0.20, "res": 0.10},
        "traits": ["domaine spectral inhabituel", "images 640×512", "temps réel ADAS"],
        "reco": "Hors distribution d'entraînement VLM : un détecteur classique fine-tuné reste solide.",
    },
    "SKU-110K": {
        "icon": "🛒", "domain": "Retail / rayonnages",
        "desc": "~150 produits par image en rayonnage — la densité maximale du domaine.",
        "difficulty": 0.66, "speed_factor": 0.85,
        "affinity": {"VLM Grounding": 1.04, "Classical": 0.94},
        "weights": {"acc": 0.50, "speed": 0.20, "track": 0.10, "res": 0.20},
        "traits": ["densité extrême", "objets quasi identiques", "NMS critique"],
        "reco": "Rayonnages denses : précision et VRAM (batch de crops) priment.",
    },
    "Global Wheat 2021": {
        "icon": "🌾", "domain": "Agriculture",
        "desc": "Épis de blé en champ, multi-continents, éclairages variables.",
        "difficulty": 0.72, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 0.95, "Classical": 1.05},
        "weights": {"acc": 0.45, "speed": 0.30, "track": 0.10, "res": 0.15},
        "traits": ["objets homogènes répétés", "texture végétale", "domain shift géo"],
        "reco": "Objets répétitifs sans sémantique riche : le détecteur spécialisé l'emporte.",
    },
    "SeaShips": {
        "icon": "🚢", "domain": "Maritime",
        "desc": "6 types de navires depuis la côte — horizon, brume, reflets.",
        "difficulty": 0.92, "speed_factor": 1.05,
        "affinity": {"VLM Grounding": 1.00, "Classical": 1.03},
        "weights": {"acc": 0.35, "speed": 0.30, "track": 0.25, "res": 0.10},
        "traits": ["grands objets lents", "fond simple", "suivi longue durée"],
        "reco": "Cibles larges et lentes : quasi tous les modèles performent, la vitesse départage.",
    },
    "iWildCam 2022": {
        "icon": "🦁", "domain": "Faune sauvage",
        "desc": "Pièges photo : espèces animales, camouflage, nuit infrarouge.",
        "difficulty": 0.76, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 1.10, "Classical": 0.88},
        "weights": {"acc": 0.50, "speed": 0.15, "track": 0.15, "res": 0.20},
        "traits": ["open-vocabulary espèces", "camouflage", "longue traîne"],
        "reco": "Espèces en longue traîne : l'open-vocabulary des VLM est décisif.",
    },
    "TT100K": {
        "icon": "🚦", "domain": "Panneaux routiers",
        "desc": "100k images, 200+ classes de panneaux chinois, souvent < 32 px.",
        "difficulty": 0.80, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 1.03, "Classical": 0.96},
        "weights": {"acc": 0.45, "speed": 0.30, "track": 0.10, "res": 0.15},
        "traits": ["petites cibles", "200+ classes", "lecture fine"],
        "reco": "Beaucoup de classes fines : la compréhension visuelle fine des VLM aide.",
    },
    "SportsMOT": {
        "icon": "🏀", "domain": "Sport / tracking",
        "desc": "Basket, foot, volley : joueurs similaires, mouvements rapides.",
        "difficulty": 0.84, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 1.00, "Classical": 1.00},
        "weights": {"acc": 0.20, "speed": 0.30, "track": 0.40, "res": 0.10},
        "traits": ["apparence homogène", "motion blur", "ID switches fréquents"],
        "reco": "Joueurs interchangeables : le couple détecteur rapide + tracker robuste gagne.",
    },
    "ICDAR 2015": {
        "icon": "🔤", "domain": "Texte en scène",
        "desc": "Texte incidentel en environnement urbain — orientations libres.",
        "difficulty": 0.70, "speed_factor": 0.95,
        "affinity": {"VLM Grounding": 1.15, "Classical": 0.75},
        "weights": {"acc": 0.55, "speed": 0.20, "track": 0.05, "res": 0.20},
        "traits": ["texte multi-orienté", "flou de mouvement", "lecture requise"],
        "reco": "Localiser et lire du texte : domaine natif des VLM, hors périmètre des détecteurs boxes.",
    },
    "TACO": {
        "icon": "♻️", "domain": "Déchets / environnement",
        "desc": "Déchets en milieu naturel : 60 catégories, formes déformées.",
        "difficulty": 0.70, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 1.12, "Classical": 0.85},
        "weights": {"acc": 0.50, "speed": 0.20, "track": 0.10, "res": 0.20},
        "traits": ["objets déformés", "open-vocabulary", "fonds naturels"],
        "reco": "Objets déformés/inédits : la généralisation open-vocabulary VLM est clé.",
    },
    "GRAZPEDWRI-DX": {
        "icon": "🦴", "domain": "Médical / radiographie",
        "desc": "Fractures pédiatriques sur radios du poignet — précision critique.",
        "difficulty": 0.65, "speed_factor": 0.90,
        "affinity": {"VLM Grounding": 1.05, "Classical": 0.92},
        "weights": {"acc": 0.60, "speed": 0.10, "track": 0.05, "res": 0.25},
        "traits": ["imagerie grayscale", "faux négatifs coûteux", "offline"],
        "reco": "Précision avant tout : chaque point de recall compte, la latence est secondaire.",
    },
    "Custom": {
        "icon": "🧪", "domain": "Personnalisé",
        "desc": "Votre propre dataset annoté (profil neutre).",
        "difficulty": 1.00, "speed_factor": 1.00,
        "affinity": {"VLM Grounding": 1.00, "Classical": 1.00},
        "weights": {"acc": 0.35, "speed": 0.30, "track": 0.20, "res": 0.15},
        "traits": ["profil neutre"],
        "reco": "Profil neutre : chargez vos annotations pour un benchmark réel.",
    },
}


def _dataset_adjusted(base: dict, ds: dict) -> dict:
    """Applique le profil dataset (difficulté, affinité, débit) aux métriques de base."""
    m = dict(base)
    q = ds["difficulty"] * ds["affinity"].get(base["category"], 1.0)
    for k in ("map50", "map50_95", "precision", "recall", "mota"):
        m[k] = round(min(base[k] * q, 0.95), 3)
    sf = ds.get("speed_factor", 1.0)
    m["fps_1080p"] = round(base["fps_1080p"] * sf, 1)
    m["fps_720p"] = round(base["fps_720p"] * sf, 1)
    m["latency_ms"] = round(base["latency_ms"] / max(sf, 0.1))
    m["id_switches_per_1k"] = max(1, round(base["id_switches_per_1k"] / max(q, 0.1)))
    return m


def _rank_models(adj: dict, weights: dict) -> list[tuple[str, float]]:
    """Score de sélection auto : pondération accuracy / vitesse / tracking / ressources."""
    names = list(adj.keys())
    if not names:
        return []
    max_fps = max(adj[n]["fps_1080p"] for n in names) or 1
    min_vram = min(adj[n]["vram_gb"] for n in names)
    max_vram = max(adj[n]["vram_gb"] for n in names)
    scores = {}
    for n in names:
        m = adj[n]
        s_acc = m["map50"]
        s_speed = m["fps_1080p"] / max_fps
        s_track = m["mota"]
        s_res = 1 - (m["vram_gb"] - min_vram) / max(max_vram - min_vram, 1)
        scores[n] = round(
            weights["acc"] * s_acc + weights["speed"] * s_speed
            + weights["track"] * s_track + weights["res"] * s_res,
            3,
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


random.seed(42)
np.random.seed(42)


def _jitter(val, pct=0.03):
    return val * (1 + random.uniform(-pct, pct))


def _gen_timeseries(n=200, base=30.0, noise=3.0, trend=0.0):
    t = np.arange(n, dtype=float)
    signal = base + trend * t / n + noise * np.random.randn(n)
    signal += 2.0 * np.sin(2 * np.pi * t / 50)
    return np.clip(signal, 0, None)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📐 Benchmark & KPIs DataOps")
st.markdown(
    "Monitoring complet des modèles de détection : **throughput, latence, précision, tracking, "
    "ressources GPU, qualité de données et coût opérationnel**."
)

# ── Sidebar : sélection benchmark ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration Benchmark")
    st.markdown("---")

    st.markdown("**Résolution de référence**")
    resolution = st.selectbox("Résolution", ["720p", "1080p", "4K"], index=1)

    st.markdown("**Dataset d'évaluation** (20 domaines)")
    dataset = st.selectbox(
        "Dataset",
        list(DATASETS.keys()),
        index=0,
        format_func=lambda d: f"{DATASETS[d]['icon']} {d} — {DATASETS[d]['domain']}",
    )
    _ds = DATASETS[dataset]
    st.caption(_ds["desc"])
    st.caption("· ".join(_ds["traits"]))

    st.markdown("**Durée simulation**")
    sim_duration = st.slider("Minutes de métriques", 5, 60, 15)

    st.markdown("---")
    st.markdown("**Modèles comparés**")
    selected_models = []
    for name in BASE_MODELS:
        if name in EN_ENTRETIEN:
            st.checkbox(f"🔧 {name} (en cours d'entretien)", False, key=f"chk_{name}",
                        disabled=True, help=EN_ENTRETIEN[name])
            continue
        if st.checkbox(name, True, key=f"chk_{name}"):
            selected_models.append(name)

    st.markdown("---")
    if st.button("🔄 Rafraîchir données", use_container_width=True):
        st.rerun()

# Métriques ajustées au profil du dataset sélectionné —
# tout le reste de la page lit ce dict.
MODELS = {name: _dataset_adjusted(BASE_MODELS[name], _ds) for name in BASE_MODELS}

# ── Section 0 : Sélection automatique du modèle ──────────────────────────────
render_section_header(
    "🤖", "Sélection automatique du modèle",
    f"Recommandation pour {_ds['icon']} {dataset} ({_ds['domain']}) — "
    f"pondération : {int(_ds['weights']['acc']*100)}% précision · "
    f"{int(_ds['weights']['speed']*100)}% vitesse · "
    f"{int(_ds['weights']['track']*100)}% tracking · "
    f"{int(_ds['weights']['res']*100)}% ressources",
)

if selected_models:
    _adj_sel = {n: MODELS[n] for n in selected_models}
    auto_ranking = _rank_models(_adj_sel, _ds["weights"])
    _best_name, _best_score = auto_ranking[0]
    _best_m = MODELS[_best_name]

    st.markdown(
        f"""<div class="model-eval-card winner">
        🥇 <strong>{_best_name}</strong> — score {_best_score:.3f} sur {_ds['icon']} <strong>{dataset}</strong><br>
        <span style="color:var(--text-secondary);font-size:0.85rem">
        mAP@0.5 ajusté {_best_m['map50']:.3f} · {_best_m['fps_1080p']:.0f} FPS (1080p) ·
        MOTA {_best_m['mota']:.3f} · {_best_m['vram_gb']} GB VRAM<br>
        💡 {_ds['reco']}
        </span></div>""",
        unsafe_allow_html=True,
    )

    podium_cols = st.columns(min(len(auto_ranking), 5))
    for i, (name, score) in enumerate(auto_ranking[:5]):
        with podium_cols[i]:
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            st.markdown(
                f"""<div class="metric-card">
                <div style="font-size:1.3rem">{medal}</div>
                <div class="metric-value" style="font-size:1.4rem">{score:.3f}</div>
                <div class="metric-label">{name}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    with st.expander("🗺️ Matrice — modèle recommandé par dataset (tous domaines)"):
        matrix_rows = []
        for ds_name, ds_cfg in DATASETS.items():
            adj_all = {n: _dataset_adjusted(BASE_MODELS[n], ds_cfg) for n in selected_models}
            rk = _rank_models(adj_all, ds_cfg["weights"])
            matrix_rows.append({
                "Dataset": f"{ds_cfg['icon']} {ds_name}",
                "Domaine": ds_cfg["domain"],
                "Modèle recommandé": rk[0][0],
                "Score": rk[0][1],
                "2ème choix": rk[1][0] if len(rk) > 1 else "—",
            })
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)
else:
    st.warning("Sélectionnez au moins un modèle dans la sidebar.")
    st.stop()

st.markdown("---")

# ── Section 1 : Vue d'ensemble ────────────────────────────────────────────────
render_section_header(
    "🏁", "Vue d'ensemble — Classement multi-critères",
    f"Résolution : {resolution} | Dataset : {_ds['icon']} {dataset} ({_ds['domain']}) — métriques ajustées au domaine",
)

fps_key = "fps_720p" if resolution == "720p" else ("fps_1080p" if resolution == "1080p" else "fps_1080p")
fps_mult = 0.4 if resolution == "4K" else 1.0

df_models = pd.DataFrame([
    {
        "Modèle": name,
        "Catégorie": m["category"],
        "Params": m["params"],
        "VRAM (GB)": m["vram_gb"],
        f"FPS ({resolution})": round(_jitter(m[fps_key] * fps_mult), 1),
        "Latence (ms)": round(_jitter(m["latency_ms"] / fps_mult)),
        "mAP@0.5": round(_jitter(m["map50"]), 3),
        "mAP@0.5:0.95": round(_jitter(m["map50_95"]), 3),
        "Precision": round(_jitter(m["precision"]), 3),
        "Recall": round(_jitter(m["recall"]), 3),
        "MOTA": round(_jitter(m["mota"]), 3),
        "ID Switches/1k": round(_jitter(m["id_switches_per_1k"])),
        "Puissance (W)": m["power_w"],
        "Modèle (MB)": m["model_size_mb"],
    }
    for name, m in MODELS.items()
    if name in selected_models
])

st.dataframe(
    df_models.style
        .highlight_max(subset=[f"FPS ({resolution})", "mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "MOTA"], color="rgba(74,222,128,0.15)")
        .highlight_min(subset=["Latence (ms)", "VRAM (GB)", "ID Switches/1k", "Puissance (W)"], color="rgba(74,222,128,0.15)"),
    use_container_width=True,
    hide_index=True,
)

# ── Section 2 : KPIs Throughput & Latence ─────────────────────────────────────
st.markdown("---")
render_section_header("⚡", "KPIs Throughput & Latence", "Ce que surveille le DataOps en production")

n_points = sim_duration * 4
ts_time = [datetime.now() - timedelta(seconds=n_points - i) for i in range(n_points)]

col_tp1, col_tp2 = st.columns(2)

with col_tp1:
    st.markdown("##### FPS temps réel (inférence pure)")
    fig_data = {}
    for name in selected_models:
        m = MODELS[name]
        base_fps = m[fps_key] * fps_mult
        fig_data[name] = _gen_timeseries(n_points, base=base_fps, noise=base_fps * 0.08)

    chart_df = pd.DataFrame(fig_data, index=ts_time[:n_points])
    st.line_chart(chart_df, height=300)

    fps_vals = {name: np.mean(vals) for name, vals in fig_data.items()}
    best_fps_name = max(fps_vals, key=fps_vals.get)
    worst_fps_name = min(fps_vals, key=fps_vals.get)

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Meilleur FPS", f"{fps_vals[best_fps_name]:.1f}", best_fps_name.split("-")[0])
    with kpi_cols[1]:
        st.metric("Pire FPS", f"{fps_vals[worst_fps_name]:.1f}", worst_fps_name.split("-")[0])
    with kpi_cols[2]:
        st.metric("FPS moyen (tous)", f"{np.mean(list(fps_vals.values())):.1f}")
    with kpi_cols[3]:
        real_time_feasible = sum(1 for v in fps_vals.values() if v >= 25)
        st.metric("Modèles temps réel (≥25fps)", f"{real_time_feasible}/{len(selected_models)}")

with col_tp2:
    st.markdown("##### Latence par frame (ms)")
    lat_data = {}
    for name in selected_models:
        m = MODELS[name]
        base_lat = m["latency_ms"] / fps_mult
        lat_data[name] = _gen_timeseries(n_points, base=base_lat, noise=base_lat * 0.12)

    lat_df = pd.DataFrame(lat_data, index=ts_time[:n_points])
    st.line_chart(lat_df, height=300)

    lat_vals = {name: np.mean(vals) for name, vals in lat_data.items()}
    p99_vals = {name: np.percentile(vals, 99) for name, vals in lat_data.items()}

    kpi_cols2 = st.columns(4)
    with kpi_cols2[0]:
        best_lat_name = min(lat_vals, key=lat_vals.get)
        st.metric("Meilleure latence moy.", f"{lat_vals[best_lat_name]:.0f}ms", best_lat_name.split("-")[0])
    with kpi_cols2[1]:
        worst_p99_name = max(p99_vals, key=p99_vals.get)
        st.metric("Pire P99", f"{p99_vals[worst_p99_name]:.0f}ms", worst_p99_name.split("-")[0])
    with kpi_cols2[2]:
        jitter_max = max(np.max(vals) - np.min(vals) for vals in lat_data.values())
        st.metric("Max jitter", f"{jitter_max:.0f}ms")
    with kpi_cols2[3]:
        sla_100ms = sum(1 for v in p99_vals.values() if v <= 100)
        st.metric("Modèles sous SLA 100ms (P99)", f"{sla_100ms}/{len(selected_models)}")

# ── Section 3 : KPIs Précision & Accuracy ─────────────────────────────────────
st.markdown("---")
render_section_header("🎯", "KPIs Précision & Accuracy", "Métriques d'évaluation standard COCO / MOT")

tab_acc1, tab_acc2, tab_acc3, tab_acc4 = st.tabs([
    "mAP & F1", "Precision-Recall", "Confiance distributions", "Par classe"
])

with tab_acc1:
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("##### mAP@0.5 vs mAP@0.5:0.95")
        map_data = []
        for name in selected_models:
            m = MODELS[name]
            map_data.append({"Modèle": name, "mAP@0.5": round(_jitter(m["map50"]), 3), "mAP@0.5:0.95": round(_jitter(m["map50_95"]), 3)})
        map_df = pd.DataFrame(map_data).set_index("Modèle")
        st.bar_chart(map_df, height=300)

    with col_a2:
        st.markdown("##### F1-Score par modèle")
        f1_data = {}
        for name in selected_models:
            m = MODELS[name]
            p, r = _jitter(m["precision"]), _jitter(m["recall"])
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_data[name] = round(f1, 3)

        f1_df = pd.DataFrame({"Modèle": list(f1_data.keys()), "F1": list(f1_data.values())}).set_index("Modèle")
        st.bar_chart(f1_df, height=300, color="#7F77DD")

    kpi_acc = st.columns(5)
    with kpi_acc[0]:
        st.metric("Meilleur mAP@0.5", f"{max(_jitter(MODELS[n]['map50']) for n in selected_models):.3f}")
    with kpi_acc[1]:
        st.metric("Meilleur mAP@0.5:0.95", f"{max(_jitter(MODELS[n]['map50_95']) for n in selected_models):.3f}")
    with kpi_acc[2]:
        st.metric("Meilleur F1", f"{max(f1_data.values()):.3f}")
    with kpi_acc[3]:
        avg_p = np.mean([_jitter(MODELS[n]['precision']) for n in selected_models])
        st.metric("Precision moyenne", f"{avg_p:.3f}")
    with kpi_acc[4]:
        avg_r = np.mean([_jitter(MODELS[n]['recall']) for n in selected_models])
        st.metric("Recall moyen", f"{avg_r:.3f}")

with tab_acc2:
    st.markdown("##### Courbes Precision-Recall simulées")

    fig_pr = go.Figure()
    for name in selected_models:
        m = MODELS[name]
        recalls = np.linspace(0, 1, 50)
        p_base = _jitter(m["precision"])
        r_base = _jitter(m["recall"])
        precisions = p_base * (1 - 0.6 * (recalls / max(r_base, 0.01)) ** 1.5)
        precisions = np.clip(precisions, 0, 1)
        fig_pr.add_trace(go.Scatter(x=recalls, y=precisions, mode="lines", name=name))

    fig_pr.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        height=400, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

with tab_acc3:
    st.markdown("##### Distribution des scores de confiance")

    fig_conf = go.Figure()
    for name in selected_models:
        m = MODELS[name]
        mu = _jitter(m["precision"], 0.05)
        sigma = 0.12 + random.random() * 0.08
        scores = np.clip(np.random.normal(mu, sigma, 2000), 0, 1)
        fig_conf.add_trace(go.Histogram(x=scores, name=name, opacity=0.6, nbinsx=30))

    fig_conf.update_layout(
        xaxis_title="Score de confiance", yaxis_title="Fréquence",
        barmode="overlay", height=350, template="plotly_dark",
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    st.markdown("##### KPIs qualité des scores")
    conf_kpis = st.columns(4)
    with conf_kpis[0]:
        st.metric("Confiance médiane (best)", f"{max(_jitter(MODELS[n]['precision'], 0.05) for n in selected_models):.3f}")
    with conf_kpis[1]:
        st.metric("Écart-type moyen", f"{np.mean([0.12 + random.random() * 0.08 for _ in selected_models]):.3f}")
    with conf_kpis[2]:
        st.metric("Taux score > 0.9", f"{random.randint(15, 35)}%")
    with conf_kpis[3]:
        st.metric("Taux score < 0.3", f"{random.randint(3, 12)}%")

with tab_acc4:
    st.markdown("##### Performance par classe d'objet (COCO-like)")
    classes = ["person", "car", "truck", "bicycle", "motorcycle", "bus", "dog", "cat", "bottle", "chair"]
    selected_model_for_class = st.selectbox("Modèle", selected_models, key="class_model")
    m = MODELS[selected_model_for_class]

    class_data = []
    for cls in classes:
        base_ap = _jitter(m["map50"], 0.15)
        cls_ap = np.clip(base_ap + random.uniform(-0.15, 0.15), 0.05, 0.95)
        class_data.append({"Classe": cls, "AP@0.5": round(cls_ap, 3)})

    class_df = pd.DataFrame(class_data).set_index("Classe")
    st.bar_chart(class_df, height=350, color="#7F77DD")

    st.markdown("##### Classes faibles (AP < 0.4) — alertes DataOps")
    weak = class_df[class_df["AP@0.5"] < 0.4]
    if len(weak) > 0:
        for cls, row in weak.iterrows():
            st.warning(f"**{cls}** — AP@0.5 = {row['AP@0.5']:.3f} — Enrichissement dataset recommandé")
    else:
        st.success("Aucune classe sous le seuil de 0.4")

# ── Section 4 : KPIs Tracking Multi-Objets ────────────────────────────────────
st.markdown("---")
render_section_header("👥", "KPIs Tracking Multi-Objets (MOT)", "Métriques MOTA / IDA — évaluation du suivi temporel")

mot_cols = st.columns(6)
for i, name in enumerate(selected_models[:6]):
    m = MODELS[name]
    with mot_cols[i % 6]:
        mota = round(_jitter(m["mota"]), 3)
        st.metric(name.split("-")[0], f"MOTA {mota}")

st.markdown("##### MOTA & ID Switches — timeline")
col_m1, col_m2 = st.columns(2)

with col_m1:
    mota_data = {}
    for name in selected_models:
        m = MODELS[name]
        mota_data[name] = _gen_timeseries(n_points, base=m["mota"], noise=0.03, trend=-0.005)
    mota_df = pd.DataFrame(mota_data, index=ts_time[:n_points])
    st.line_chart(mota_df, height=280)

with col_m2:
    ids_data = {}
    for name in selected_models:
        m = MODELS[name]
        ids_data[name] = _gen_timeseries(n_points, base=m["id_switches_per_1k"], noise=3.0)
    ids_df = pd.DataFrame(ids_data, index=ts_time[:n_points])
    st.line_chart(ids_df, height=280)

st.markdown("##### KPIs tracking détaillés")
track_kpis = st.columns(5)
with track_kpis[0]:
    best_mota = max(selected_models, key=lambda n: MODELS[n]["mota"])
    st.metric("Meilleur MOTA", f"{MODELS[best_mota]['mota']:.3f}", best_mota.split("-")[0])
with track_kpis[1]:
    best_ids = min(selected_models, key=lambda n: MODELS[n]["id_switches_per_1k"])
    st.metric("Moins d'ID switches", f"{MODELS[best_ids]['id_switches_per_1k']}/1k frames", best_ids.split("-")[0])
with track_kpis[2]:
    st.metric("IDF1 moyen", f"{np.mean([_jitter(MODELS[n]['mota'] + 0.05) for n in selected_models]):.3f}")
with track_kpis[3]:
    st.metric("Frag. moyenne", f"{random.randint(45, 120)}")
with track_kpis[4]:
    st.metric("Track length médian", f"{random.randint(80, 200)} frames")

# ── Section 5 : KPIs Ressources GPU ──────────────────────────────────────────
st.markdown("---")
render_section_header("🖥️", "KPIs Ressources GPU & Infrastructure", "Monitoring RTX 5090 — VRAM, puissance, température")

col_g1, col_g2, col_g3 = st.columns(3)

with col_g1:
    st.markdown("##### VRAM allouée (GB)")
    vram_data = {}
    for name in selected_models:
        m = MODELS[name]
        vram_data[name] = _gen_timeseries(n_points, base=m["vram_gb"], noise=0.5)
    vram_df = pd.DataFrame(vram_data, index=ts_time[:n_points])
    st.line_chart(vram_df, height=250)

with col_g2:
    st.markdown("##### Puissance (W)")
    power_data = {}
    for name in selected_models:
        m = MODELS[name]
        power_data[name] = _gen_timeseries(n_points, base=m["power_w"], noise=15)
    power_df = pd.DataFrame(power_data, index=ts_time[:n_points])
    st.line_chart(power_df, height=250)

with col_g3:
    st.markdown("##### Température GPU (°C)")
    temp_data = {}
    for name in selected_models:
        m = MODELS[name]
        base_temp = 62 + (m["power_w"] - 195) / 200 * 15
        temp_data[name] = _gen_timeseries(n_points, base=base_temp, noise=2.5)
    temp_df = pd.DataFrame(temp_data, index=ts_time[:n_points])
    st.line_chart(temp_df, height=250)

gpu_kpis = st.columns(6)
with gpu_kpis[0]:
    total_vram = sum(MODELS[n]["vram_gb"] for n in selected_models)
    st.metric("VRAM totale requise", f"{total_vram} GB", "32 GB dispo")
with gpu_kpis[1]:
    max_power = max(MODELS[n]["power_w"] for n in selected_models)
    st.metric("Pic puissance", f"{max_power} W")
with gpu_kpis[2]:
    avg_temp = np.mean([62 + (MODELS[n]["power_w"] - 195) / 200 * 15 for n in selected_models])
    st.metric("Temp. moyenne", f"{avg_temp:.0f}°C", delta="OK" if avg_temp < 80 else "HOT")
with gpu_kpis[3]:
    st.metric("GPU util. moyenne", f"{random.randint(75, 95)}%")
with gpu_kpis[4]:
    st.metric("GPU mem. util.", f"{random.randint(60, 90)}%", f"{total_vram}/32 GB")
with gpu_kpis[5]:
    st.metric("OOM errors (24h)", "0", delta="stable")

# ── Section 6 : KPIs Opérationnels / Pipeline ─────────────────────────────────
st.markdown("---")
render_section_header("🔧", "KPIs Opérationnels — Pipeline & Queue", "Ce que le DataOps surveille au quotidien")

col_o1, col_o2 = st.columns(2)

with col_o1:
    st.markdown("##### Profondeur queue Celery")
    queue_data = {
        "detection_queue": _gen_timeseries(n_points, base=3, noise=4),
        "vlm_queue": _gen_timeseries(n_points, base=1, noise=2),
        "clustering_queue": _gen_timeseries(n_points, base=0.5, noise=1),
    }
    queue_df = pd.DataFrame(queue_data, index=ts_time[:n_points])
    st.line_chart(queue_df, height=250)

with col_o2:
    st.markdown("##### Temps de traitement par tâche (s)")
    task_data = {
        "decode_video": _gen_timeseries(n_points // 2, base=2.1, noise=0.5),
        "preprocess": _gen_timeseries(n_points // 2, base=0.8, noise=0.2),
        "inference": _gen_timeseries(n_points // 2, base=1.2, noise=0.4),
        "postprocess": _gen_timeseries(n_points // 2, base=0.3, noise=0.1),
    }
    task_df = pd.DataFrame(task_data, index=ts_time[:n_points // 2])
    st.line_chart(task_df, height=250)

ops_kpis = st.columns(6)
with ops_kpis[0]:
    st.metric("Tasks/min", f"{random.randint(45, 85)}")
with ops_kpis[1]:
    st.metric("Queue depth moy.", f"{random.randint(2, 8)}")
with ops_kpis[2]:
    error_rate = random.uniform(0.001, 0.02)
    st.metric("Error rate", f"{error_rate:.2%}", delta="OK" if error_rate < 0.01 else "ALERT")
with ops_kpis[3]:
    st.metric("P50 task duration", f"{random.uniform(1.5, 3.5):.1f}s")
with ops_kpis[4]:
    st.metric("P95 task duration", f"{random.uniform(5, 12):.1f}s")
with ops_kpis[5]:
    st.metric("Retry rate", f"{random.uniform(0.5, 3):.1f}%")

# ── Section 7 : KPIs Qualité de Données ──────────────────────────────────────
st.markdown("---")
render_section_header("📊", "KPIs Qualité de Données & Data Drift", "Détection de dérive et anomalies dans les flux")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown("##### Frame drop rate")
    drop_data = _gen_timeseries(n_points, base=0.5, noise=0.8)
    drop_df = pd.DataFrame({"frame_drop_pct": drop_data}, index=ts_time[:n_points])
    st.line_chart(drop_df, height=220)

with col_d2:
    st.markdown("##### Distribution IoU (ground truth vs prédit)")
    iou_samples = np.clip(np.random.beta(5, 2, 1000), 0, 1)
    fig_iou = go.Figure(go.Histogram(x=iou_samples, nbinsx=25, marker_color="#7F77DD"))
    fig_iou.update_layout(xaxis_title="IoU", yaxis_title="Count", height=220, template="plotly_dark")
    st.plotly_chart(fig_iou, use_container_width=True)

with col_d3:
    st.markdown("##### Drift score (KL divergence)")
    drift_data = _gen_timeseries(n_points, base=0.05, noise=0.03, trend=0.02)
    drift_df = pd.DataFrame({"kl_divergence": drift_data}, index=ts_time[:n_points])
    st.line_chart(drift_df, height=220)

data_kpis = st.columns(6)
with data_kpis[0]:
    st.metric("Frame drop rate", f"{np.mean(drop_data):.2f}%")
with data_kpis[1]:
    st.metric("IoU médian", f"{np.median(iou_samples):.3f}")
with data_kpis[2]:
    st.metric("IoU > 0.5 ratio", f"{np.mean(iou_samples > 0.5):.1%}")
with data_kpis[3]:
    drift_now = drift_data[-1]
    st.metric("Drift KL (actuel)", f"{drift_now:.4f}", delta="OK" if drift_now < 0.1 else "DRIFT")
with data_kpis[4]:
    st.metric("Résolution effective", f"{random.choice(['1920x1080', '1280x720'])}")
with data_kpis[5]:
    st.metric("Occlusion rate", f"{random.randint(5, 20)}%")

# ── Section 8 : KPIs Coût & Efficacité ────────────────────────────────────────
st.markdown("---")
render_section_header("💰", "KPIs Coût & Efficacité", "Optimisation coût/performance pour la production")

cost_data = []
for name in selected_models:
    m = MODELS[name]
    fps = m[fps_key] * fps_mult
    power_kwh = m["power_w"] / 1000
    cost_per_hour = power_kwh * 0.15
    frames_per_hour = fps * 3600
    cost_per_1k_frames = cost_per_hour / (frames_per_hour / 1000) if frames_per_hour > 0 else 999
    efficiency = _jitter(m["map50"]) / max(m["power_w"] / 100, 0.01)

    cost_data.append({
        "Modèle": name,
        "Coût/heure (€)": round(cost_per_hour, 3),
        "Frames/heure": int(frames_per_hour),
        "Coût/1k frames (€)": round(cost_per_1k_frames, 5),
        "Efficacité (mAP/W)": round(efficiency, 3),
        "Coût/vidéo 1h (€)": round(cost_per_hour, 2),
    })

cost_df = pd.DataFrame(cost_data).set_index("Modèle")

st.dataframe(
    cost_df.style
        .highlight_min(subset=["Coût/1k frames (€)", "Coût/heure (€)"], color="rgba(74,222,128,0.15)")
        .highlight_max(subset=["Efficacité (mAP/W)", "Frames/heure"], color="rgba(74,222,128,0.15)"),
    use_container_width=True,
)

col_c1, col_c2 = st.columns(2)

with col_c1:
    st.markdown("##### Coût/1k frames vs Efficacité")
    fig_eff = go.Figure()
    for _, row in cost_df.reset_index().iterrows():
        fig_eff.add_trace(go.Scatter(
            x=[row["Coût/1k frames (€)"]],
            y=[cost_df.loc[row["Modèle"], "Efficacité (mAP/W)"]],
            mode="markers+text",
            name=row["Modèle"],
            text=[row["Modèle"].split("-")[0]],
            textposition="top center",
            marker=dict(size=max(row["Frames/heure"] / 5000, 10)),
        ))
    fig_eff.update_layout(
        xaxis_title="Coût/1k frames (€)",
        yaxis_title="Efficacité (mAP/W)",
        height=350, template="plotly_dark",
    )
    st.plotly_chart(fig_eff, use_container_width=True)

with col_c2:
    st.markdown("##### VRAM vs FPS — Pareto frontier")
    fig_pareto = go.Figure()
    for name in selected_models:
        m = MODELS[name]
        fig_pareto.add_trace(go.Scatter(
            x=[m["vram_gb"]],
            y=[m[fps_key] * fps_mult],
            mode="markers+text",
            name=name,
            text=[name.split("-")[0]],
            textposition="top center",
            marker=dict(size=m["map50"] * 40, color="#7F77DD"),
        ))
    fig_pareto.update_layout(
        xaxis_title="VRAM (GB)",
        yaxis_title=f"FPS ({resolution})",
        height=350, template="plotly_dark",
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

# ── Section 9 : Score composite ──────────────────────────────────────────────
st.markdown("---")
render_section_header("🏆", "Score Composite — Ranking DataOps", "Score pondéré : 30% précision + 25% vitesse + 20% coût + 15% tracking + 10% ressources")

scores = {}
for name in selected_models:
    m = MODELS[name]
    fps = m[fps_key] * fps_mult
    max_fps = max(MODELS[n][fps_key] * fps_mult for n in selected_models)

    s_precision = m["map50"]
    s_speed = fps / max_fps
    s_cost = 1 - (m["power_w"] - min(MODELS[n]["power_w"] for n in selected_models)) / max(
        max(MODELS[n]["power_w"] for n in selected_models) - min(MODELS[n]["power_w"] for n in selected_models), 1
    )
    s_tracking = m["mota"]
    s_resource = 1 - (m["vram_gb"] - min(MODELS[n]["vram_gb"] for n in selected_models)) / max(
        max(MODELS[n]["vram_gb"] for n in selected_models) - min(MODELS[n]["vram_gb"] for n in selected_models), 1
    )

    composite = (
        0.30 * s_precision +
        0.25 * s_speed +
        0.20 * s_cost +
        0.15 * s_tracking +
        0.10 * s_resource
    )
    scores[name] = round(composite, 3)

ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

rank_cols = st.columns(len(ranking))
for i, (name, score) in enumerate(ranking):
    with rank_cols[i]:
        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"#{i+1}"
        st.markdown(f"""
        <div class="metric-card" style="border-color: {'#4ade80' if i == 0 else 'var(--border)'}">
            <div style="font-size:1.5rem">{medal}</div>
            <div class="metric-value">{score:.3f}</div>
            <div class="metric-label">{name}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Section 10 : Recommandations DataOps ──────────────────────────────────────
st.markdown("---")
render_section_header("💡", "Recommandations DataOps", "Actions prioritaires basées sur les KPIs")

best_model = ranking[0][0]

recs = [
    (f"**Meilleur compromis** : {best_model} (score {ranking[0][1]:.3f})", "success"),
]

for name in selected_models:
    m = MODELS[name]
    fps = m[fps_key] * fps_mult
    if fps < 25:
        recs.append((f"**{name}** : sous le seuil temps réel ({fps:.1f} FPS) — envisager batch inference ou downscale", "warning"))
    if m["vram_gb"] > 24:
        recs.append((f"**{name}** : VRAM > 24GB — quantization INT8/INT4 recommandée", "warning"))
    if m["mota"] < 0.45:
        recs.append((f"**{name}** : MOTA faible ({m['mota']:.3f}) — améliorer le tracker (BoT-SORT > ByteTrack)", "warning"))

recs.append(("**Monitoring continu** : activer alertes P99 latence > 200ms, VRAM > 90%, error rate > 1%", "info"))
recs.append(("**Data drift** : recalibrer modèle si KL divergence > 0.1 sur 1h glissante", "info"))
recs.append(("**A/B testing** : déployer les 2 meilleurs modèles en shadow mode pendant 48h", "info"))

for rec, level in recs:
    if level == "success":
        st.success(rec)
    elif level == "warning":
        st.warning(rec)
    else:
        st.info(rec)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:var(--text-secondary);font-size:0.85rem">
    <p>Benchmark DataOps — Modèles de détection d'objets en flux vidéo</p>
    <p>Données simulées basées sur benchmarks publics — 20 datasets multi-domaines — Profil RTX 5090 32GB</p>
</div>
""", unsafe_allow_html=True)
