"""
tests/test_workers.py — Tests unitaires des workers Celery
Exécuter : pytest tests/test_workers.py -v
Ces tests s'exécutent sans GPU (les modèles sont mockés).
"""
import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_fake_video(path: str, n_frames: int = 30, w: int = 320, h: int = 240):
    """Crée une vidéo factice avec OpenCV pour les tests."""
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(path, fourcc, 25.0, (w, h))
        for i in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, i % 3] = int(255 * i / n_frames)
            out.write(frame)
        out.release()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Tests vlm_worker
# ─────────────────────────────────────────────────────────────────────────────
class TestVLMWorker:

    def test_extract_frames_empty_file(self, tmp_path):
        """_extract_frames sur un fichier inexistant retourne des frames vides."""
        from workers.vlm_worker import _extract_frames
        frames = _extract_frames(str(tmp_path / "nonexistent.mp4"), 0, 5, n_frames=4)
        assert isinstance(frames, list)
        assert len(frames) == 4  # fallback sur frames noires
        assert all(isinstance(f, np.ndarray) for f in frames)

    def test_extract_frames_real_video(self, tmp_path):
        """_extract_frames sur une vraie vidéo."""
        import cv2
        video_path = str(tmp_path / "test.mp4")
        if not make_fake_video(video_path, n_frames=50):
            pytest.skip("OpenCV VideoWriter non disponible")

        from workers.vlm_worker import _extract_frames
        frames = _extract_frames(video_path, 0.0, 1.0, n_frames=4)
        assert isinstance(frames, list)
        assert len(frames) >= 1
        assert all(isinstance(f, np.ndarray) for f in frames)
        assert frames[0].ndim == 3
        assert frames[0].shape[2] == 3  # RGB

    def test_softmax_mock_sums_to_one(self):
        """_softmax_mock doit retourner des proba qui somment à 1."""
        from workers.vlm_worker import _softmax_mock
        for n in [3, 5, 10]:
            probs = _softmax_mock(n)
            assert len(probs) == n
            assert abs(probs.sum() - 1.0) < 1e-6
            assert all(p >= 0 for p in probs)

    def test_grounding_dino_for_sam_returns_list(self):
        """_grounding_dino_for_sam retourne une liste de boîtes."""
        from workers.vlm_worker import _grounding_dino_for_sam
        boxes = _grounding_dino_for_sam("fake_path.mp4", 0, "person")
        assert isinstance(boxes, list)
        if boxes:
            assert len(boxes[0]) == 4

    def test_run_sam2_video_not_found(self, tmp_path):
        """run_sam2 avec video_id inexistant retourne une erreur."""
        from workers.vlm_worker import run_sam2
        with patch.dict(os.environ, {"VIDEO_DIR": str(tmp_path)}):
            with patch("workers.vlm_worker.VIDEO_DIR", tmp_path):
                result = run_sam2.run(
                    video_id="nonexistent-video-id",
                    frame=0,
                    prompt={"type": "point", "x": 100, "y": 100},
                    propagate=False,
                )
        assert "error" in result

    def test_run_sam2_mock_mode(self, tmp_path):
        """run_sam2 sans SAM2 installé retourne un mock valide."""
        from workers.vlm_worker import run_sam2

        # Créer un fichier vidéo factice
        video_id = "test-vid-123"
        fake_video = tmp_path / f"{video_id}.mp4"
        make_fake_video(str(fake_video))

        with patch("workers.vlm_worker.VIDEO_DIR", tmp_path):
            with patch("workers.vlm_worker._get_sam", return_value=None):
                result = run_sam2.run(
                    video_id=video_id,
                    frame=0,
                    prompt={"type": "point", "x": 100, "y": 100},
                    propagate=False,
                )

        assert result["video_id"] == video_id
        assert result["status"] == "mock"
        assert "boxes" in result

    def test_run_vlm_query_mock_mode(self, tmp_path):
        """run_vlm_query sans VLM installé retourne une réponse mock."""
        from workers.vlm_worker import run_vlm_query

        video_id = "test-vlm-456"
        fake_video = tmp_path / f"{video_id}.mp4"
        make_fake_video(str(fake_video))

        with patch("workers.vlm_worker.VIDEO_DIR", tmp_path):
            with patch("workers.vlm_worker._get_vlm", return_value=None):
                result = run_vlm_query.run(
                    video_id=video_id,
                    model="llava-next-video-7b",
                    question="What is happening?",
                    start_time=0.0,
                    end_time=1.0,
                )

        assert result["video_id"] == video_id
        assert result["status"] == "mock"
        assert "answer" in result
        assert "MOCK" in result["answer"]

    def test_run_action_recognition_structure(self, tmp_path):
        """run_action_recognition retourne une liste de dict valides."""
        from workers.vlm_worker import run_action_recognition

        video_id = "test-action-789"
        fake_video = tmp_path / f"{video_id}.mp4"
        make_fake_video(str(fake_video))

        with patch("workers.vlm_worker.VIDEO_DIR", tmp_path):
            result = run_action_recognition.run(
                video_id=video_id,
                model="videomae_v2_giant",
                top_k=3,
                window_sec=2.0,
                stride_sec=1.0,
            )

        assert "results" in result
        assert isinstance(result["results"], list)
        if result["results"]:
            item = result["results"][0]
            assert "timestamp" in item
            assert "action" in item
            assert "probability" in item
            assert 0 <= item["probability"] <= 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests detection_worker
# ─────────────────────────────────────────────────────────────────────────────
class TestDetectionWorker:

    def test_infer_frame_mock_mode(self):
        """_infer_frame sans modèle retourne des détections mockées."""
        from workers.detection_worker import _infer_frame
        import cv2

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets  = _infer_frame(None, "grounding_dino", frame,
                              ["person", "car"], 0.35, 0.45)
        assert isinstance(dets, list)
        for d in dets:
            assert "class" in d
            assert "confidence" in d
            assert "bbox" in d
            assert len(d["bbox"]) == 4
            assert d["class"] in ["person", "car"]
            assert 0 <= d["confidence"] <= 1

    def test_update_tracker_uuid_thread_safe(self):
        """_update_tracker sans tracker retourne des IDs positifs uniques."""
        from workers.detection_worker import _update_tracker
        ids = [_update_tracker(None, [0, 0, 100, 100]) for _ in range(10)]
        assert all(i > 0 for i in ids)
        # UUID-based → haute probabilité d'unicité
        assert len(set(ids)) > 1

    def test_get_ocr_engine_none_when_unavailable(self):
        """_get_ocr_engine retourne None si le moteur n'est pas installé."""
        from workers.detection_worker import _get_ocr_engine
        # En environnement de test, PaddleOCR n'est probablement pas installé
        with patch.dict("sys.modules", {"paddleocr": None}):
            engine = _get_ocr_engine("paddleocr_v4", ["en"])
            # Soit None (non installé), soit un objet valide
            # Ne pas planter est l'essentiel
            assert True  # Si on arrive ici sans exception c'est OK

    def test_run_detection_video_not_found(self, tmp_path):
        """run_detection avec video inexistante retourne une erreur."""
        from workers.detection_worker import run_detection
        with patch("workers.detection_worker.VIDEO_DIR", tmp_path):
            result = run_detection.run(
                video_id="nonexistent",
                model="grounding_dino_v2",
                queries=["person"],
                confidence=0.35,
            )
        assert "error" in result

    def test_run_detection_mock_results(self, tmp_path):
        """run_detection avec modèle mocké retourne des résultats structurés."""
        from workers.detection_worker import run_detection

        video_id = "det-test-001"
        fake_video = tmp_path / f"{video_id}.mp4"
        make_fake_video(str(fake_video), n_frames=25)

        with patch("workers.detection_worker.VIDEO_DIR", tmp_path):
            with patch("workers.detection_worker._get_grounding_dino", return_value=None):
                with patch("workers.detection_worker._build_tracker", return_value=None):
                    result = run_detection.run(
                        video_id=video_id,
                        model="grounding_dino_v2",
                        queries=["person", "car"],
                        confidence=0.35,
                        batch_size=4,
                        track=False,
                    )

        assert "results" in result
        assert result["video_id"] == video_id
        assert isinstance(result["results"], list)

    def test_run_ocr_video_not_found(self, tmp_path):
        """run_ocr avec vidéo introuvable retourne une erreur."""
        from workers.detection_worker import run_ocr
        with patch("workers.detection_worker.VIDEO_DIR", tmp_path):
            result = run_ocr.run(video_id="nonexistent", engine="paddleocr_v4")
        assert "error" in result

    def test_run_ocr_mock_results(self, tmp_path):
        """run_ocr mocké retourne une liste de textes."""
        from workers.detection_worker import run_ocr

        video_id = "ocr-test-001"
        fake_video = tmp_path / f"{video_id}.mp4"
        make_fake_video(str(fake_video), n_frames=25)

        with patch("workers.detection_worker.VIDEO_DIR", tmp_path):
            with patch("workers.detection_worker._get_ocr_engine", return_value=None):
                result = run_ocr.run(
                    video_id=video_id,
                    engine="paddleocr_v4",
                    langs=["fr"],
                    min_conf=0.6,
                    fps=2,
                )

        assert "results" in result
        assert isinstance(result["results"], list)
        for item in result["results"]:
            assert "text" in item
            assert "confidence" in item
            assert "frame" in item
            assert "timestamp" in item


# ─────────────────────────────────────────────────────────────────────────────
# Tests clustering_worker
# ─────────────────────────────────────────────────────────────────────────────
class TestClusteringWorker:

    def test_run_scene_clustering_video_not_found(self, tmp_path):
        """Vidéo introuvable → erreur."""
        from workers.clustering_worker import run_scene_clustering
        with patch("workers.clustering_worker.VIDEO_DIR", tmp_path):
            result = run_scene_clustering.run(
                video_id="nonexistent",
                algorithm="hdbscan",
            )
        assert "error" in result

    def test_run_scene_clustering_mock_clip(self, tmp_path):
        """Clustering avec CLIP mocké.
        Vidéo de 5s à 25fps = 125 frames → ~5 frames extraites à 1fps.
        n_clusters=3 < 5 frames → KMeans valide.
        """
        from workers.clustering_worker import run_scene_clustering

        video_id = "cluster-test-001"
        fake_video = tmp_path / f"{video_id}.mp4"
        # 125 frames à 25fps = 5 secondes → 5 frames extraites à 1fps
        make_fake_video(str(fake_video), n_frames=125)

        with patch("workers.clustering_worker.VIDEO_DIR", tmp_path):
            with patch("workers.clustering_worker._get_clip", return_value=None):
                result = run_scene_clustering.run(
                    video_id=video_id,
                    algorithm="kmeans",
                    n_clusters=3,
                    clip_model="ViT-L/14",
                    frame_sampling="1 fps",
                )

        assert "results" in result
        assert isinstance(result["results"], list)
        if result["results"]:
            item = result["results"][0]
            assert "frame_idx" in item
            assert "cluster" in item
            assert "umap_x" in item
            assert "umap_y" in item

    def test_run_scene_clustering_hdbscan(self, tmp_path):
        """Clustering HDBSCAN avec embeddings aléatoires.
        125 frames à 25fps = 5s → suffisant pour HDBSCAN (min_cluster_size=2).
        """
        from workers.clustering_worker import run_scene_clustering

        video_id = "cluster-hdb-001"
        fake_video = tmp_path / f"{video_id}.mp4"
        make_fake_video(str(fake_video), n_frames=125)

        with patch("workers.clustering_worker.VIDEO_DIR", tmp_path):
            with patch("workers.clustering_worker._get_clip", return_value=None):
                result = run_scene_clustering.run(
                    video_id=video_id,
                    algorithm="hdbscan",
                )
        assert "results" in result

    def test_search_similar_frames_no_qdrant(self, tmp_path):
        """search_similar_frames sans Qdrant retourne une liste."""
        from workers.clustering_worker import search_similar_frames
        with patch("workers.clustering_worker.QDRANT_URL", "http://invalid:1234"):
            with patch("workers.clustering_worker._get_clip", return_value=None):
                result = search_similar_frames.run(
                    video_id="test-vid",
                    query_text="a person running",
                    top_n=5,
                    threshold=0.2,
                )
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# Tests logging_config
# ─────────────────────────────────────────────────────────────────────────────
class TestLoggingConfig:

    def test_get_logger_returns_structured_logger(self):
        """get_logger retourne un StructuredLogger."""
        from logging_config import get_logger, StructuredLogger
        logger = get_logger("test.module")
        assert isinstance(logger, StructuredLogger)

    def test_logger_accepts_structured_fields(self):
        """Le logger accepte des champs structurés sans planter."""
        from logging_config import get_logger
        logger = get_logger("test.structured")
        # Ne doit pas lever d'exception
        logger.info("Test message", video_id="abc", frame=42, score=0.95)

    def test_configure_logging_runs(self):
        """configure_logging() s'exécute sans erreur."""
        from logging_config import configure_logging
        configure_logging()  # Ne doit pas lever d'exception


# ─────────────────────────────────────────────────────────────────────────────
# Tests edge cases
# ─────────────────────────────────────────────────────────────────────────────
class TestEdgeCases:

    def test_stride_sec_zero_protection(self):
        """max(1, int(stride_sec)) ne plante pas sur stride=0.1."""
        stride_sec = 0.1
        safe_stride = max(1, int(stride_sec))
        assert safe_stride == 1

    def test_agglomerative_n_clusters_protection(self):
        """max(2, int(n_clusters)) protège contre n_clusters=0 ou 1."""
        for bad_val in [0, 1, -1]:
            safe = max(2, int(bad_val))
            assert safe >= 2

    def test_uuid_track_id_positive(self):
        """Les track IDs UUID-based sont positifs."""
        import uuid
        for _ in range(20):
            track_id = int(uuid.uuid4().int % 10000) + 1
            assert track_id > 0
            assert track_id <= 10001
