"""
tests/conftest.py — Configuration pytest et fixtures globales
"""
import os
import sys
import pytest
import tempfile
import numpy as np

# Ajouter le répertoire api au PYTHONPATH pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

# Variables d'environnement pour les tests (sans vrai Redis / GPU)
os.environ.setdefault("REDIS_URL",    "redis://localhost:6379/0")
os.environ.setdefault("VIDEO_DIR",    tempfile.gettempdir())
os.environ.setdefault("MODELS_PATH",  tempfile.gettempdir())
os.environ.setdefault("QDRANT_URL",   "http://localhost:6333")
os.environ.setdefault("LOG_FORMAT",   "text")  # Logs lisibles en CI
os.environ.setdefault("LOG_LEVEL",    "WARNING")  # Réduire le bruit


# ── Fixtures globales ──────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def tmp_video_dir(tmp_path_factory):
    """Répertoire temporaire pour les vidéos de test (partagé sur la session)."""
    return tmp_path_factory.mktemp("videos")


@pytest.fixture
def fake_frame():
    """Frame OpenCV factice (H×W×3 BGR)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def fake_embeddings():
    """Embeddings CLIP factices normalisés."""
    embs = np.random.randn(20, 768).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


# Configuration pytest-asyncio
pytest_plugins = ("pytest_asyncio",)
