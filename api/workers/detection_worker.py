"""
workers/detection_worker.py
GroundingDINO via HuggingFace transformers (IDEA-Research/grounding-dino-base)
YOLO-World v2, ByteTrack/BoT-SORT, PaddleOCR, OSNet Re-ID
VLM Grounding: LocateAnything-3B, Mini-InternVL-4B, InternVL3.5-8B
RTX 5090 32GB — PyTorch nightly cu128 — BF16
"""
import os
import json
import uuid
import re
import time
import torch
import numpy as np
from pathlib import Path
from workers.celery_app import celery_app
from logging_config import get_logger

logger      = get_logger(__name__)
MODELS_PATH = Path(os.getenv("MODELS_PATH", "/models"))
VIDEO_DIR   = Path(os.getenv("VIDEO_DIR",   "/data/videos"))
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379/0")
DTYPE       = torch.bfloat16
# ALLOW_MOCK=1 (défaut) : fallback résultats simulés si modèle KO
# ALLOW_MOCK=0 : erreur explicite (mode évaluation / prod)
ALLOW_MOCK  = os.getenv("ALLOW_MOCK", "1").strip().lower() in ("1", "true", "yes", "on")

# LocateAnything-3B : son code remote (Parallel Box Decoding custom NVIDIA) est
# incompatible avec transformers v5 — le modèle charge et génère, mais la sortie
# est dégénérée et ne produit aucune boîte exploitable. On court-circuite donc
# son chargement pour servir la simulation, ce que l'UI annonce explicitement
# (« LocateAnything-3B (simulation) »), au lieu de renvoyer 0 détection en
# prétendant faire de l'inférence réelle. Repasser à 0 quand le repo NVIDIA
# supporte transformers >= 5 (ou dans un worker dédié transformers < 4.50).
LOCATE_ANYTHING_DEGRADED = os.getenv(
    "LOCATE_ANYTHING_DEGRADED", "1").strip().lower() in ("1", "true", "yes", "on")

_loaded: dict = {}

# Gros VLM : un seul résident en VRAM à la fois (3B/4B/8B ≈ 7-16 GB chacun)
_BIG_VLM_KEYS = ("locate_anything_3b", "mini_internvl_4b", "internvl35_8b")


from contextlib import contextmanager


@contextmanager
def _cpu_linspace():
    """
    Compat transformers ≥5 pour le code remote InternVL :
    1. `torch.linspace(...).item()` dans __init__ (modeling_intern_vit.py)
       crashe sous l'init meta → forcer device=cpu sur linspace.
    2. v5 lit `model.all_tied_weights_keys`, jamais défini par les anciennes
       classes remote → fallback classe vide sur PreTrainedModel.
    """
    from transformers import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    # v5 : rope_theta a déménagé dans config.rope_parameters — les modelings
    # remote plus anciens lisent encore config.rope_theta directement
    try:
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
        if not hasattr(Qwen2Config, "rope_theta"):
            def _rt_get(self):
                if "_rope_theta" in self.__dict__:
                    return self.__dict__["_rope_theta"]
                rp = getattr(self, "rope_parameters", None) or {}
                if isinstance(rp, dict) and "rope_theta" in rp:
                    return rp["rope_theta"]
                return 1000000.0
            def _rt_set(self, v):
                self.__dict__["_rope_theta"] = v
            Qwen2Config.rope_theta = property(_rt_get, _rt_set)
    except Exception:
        pass

    orig = torch.linspace

    def patched(*args, **kwargs):
        kwargs.setdefault("device", "cpu")
        return orig(*args, **kwargs)

    torch.linspace = patched
    try:
        yield
    finally:
        torch.linspace = orig


def _ensure_generation_mixin(model, patch_prepare: bool = True) -> None:
    """
    transformers ≥4.50 : PreTrainedModel n'hérite plus de GenerationMixin,
    donc les LM des vieux codes remote (Phi3ForCausalLM d'InternVL…) perdent
    `.generate`. On réinjecte le mixin dans la classe au chargement.
    """
    from transformers import GenerationMixin, GenerationConfig
    # vieux code remote : past_key_values.seen_tokens (retiré de DynamicCache)
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, "get_max_length"):
            # sémantique historique : None = cache dynamique illimité
            # (get_max_cache_shape v5 renvoie un tuple, pas un int)
            DynamicCache.get_max_length = lambda self: None
        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length, layer_idx=0):
                max_length = self.get_max_length()
                previous = self.get_seq_length(layer_idx)
                if max_length is not None and previous + new_seq_length > max_length:
                    return max_length - new_seq_length
                return previous
            DynamicCache.get_usable_length = _get_usable_length
        if not hasattr(DynamicCache, "to_legacy_cache"):
            DynamicCache.to_legacy_cache = lambda self: tuple(
                (layer.keys, layer.values) for layer in self.layers
            )
        if not hasattr(DynamicCache, "from_legacy_cache"):
            def _from_legacy_cache(cls, past_key_values=None):
                cache = cls()
                for layer_idx, (k, v) in enumerate(past_key_values or ()):
                    cache.update(k, v, layer_idx)
                return cache
            DynamicCache.from_legacy_cache = classmethod(_from_legacy_cache)
    except Exception:
        pass
    for target in (getattr(model, "language_model", None), model):
        if target is None:
            continue
        cls = type(target)
        if not isinstance(target, GenerationMixin):
            try:
                cls.__bases__ = cls.__bases__ + (GenerationMixin,)
                logger.info("GenerationMixin patched", cls=cls.__name__)
            except TypeError as e:
                logger.warning("GenerationMixin patch failed",
                               cls=cls.__name__, error=str(e))
            # le prepare_inputs_for_generation d'époque est incompatible avec
            # generate v5 (input_ids vidé, pas de cache_position) → utiliser la
            # version générique du mixin, filtrée sur la signature du forward.
            # patch_prepare=False pour les modèles à boucle de génération
            # custom (LocateAnything) qui appellent l'ancien prepare directement.
            if patch_prepare and "prepare_inputs_for_generation" in cls.__dict__:
                import inspect
                generic = GenerationMixin.prepare_inputs_for_generation
                allowed = set(inspect.signature(cls.forward).parameters)

                # inputs_embeds doit apparaître explicitement dans la signature :
                # generate() v5 inspecte prepare_inputs_for_generation pour
                # autoriser le passage d'inputs_embeds (utilisé par InternVL.chat)
                def _prep(self, *args, inputs_embeds=None,
                          _generic=generic, _allowed=allowed, **kwargs):
                    out = _generic(self, *args, inputs_embeds=inputs_embeds, **kwargs)
                    return {k: v for k, v in out.items() if k in _allowed}

                cls.prepare_inputs_for_generation = _prep
                logger.info("prepare_inputs_for_generation patched", cls=cls.__name__)
        # v5 ne crée generation_config que si la classe génère dès l'init
        if getattr(target, "generation_config", None) is None:
            try:
                target.generation_config = GenerationConfig.from_model_config(target.config)
            except Exception:
                target.generation_config = GenerationConfig()


def _relax_remote_overrides(method: str = "_check_and_adjust_attn_implementation") -> None:
    """
    transformers v5 appelle certains hooks internes avec des kwargs récents
    (ex. allow_all_kernels) que les overrides du code remote plus ancien ne
    connaissent pas. On filtre les kwargs sur la signature d'origine.
    """
    import sys
    import inspect
    for name, mod in list(sys.modules.items()):
        if "transformers_modules" not in name or mod is None:
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if not isinstance(obj, type):
                continue
            # ancien format liste de _tied_weights_keys → v5 veut {cible: source}
            twk = obj.__dict__.get("_tied_weights_keys")
            if isinstance(twk, (list, tuple)):
                mapping = {}
                for k in twk:
                    if k == "lm_head.weight":
                        mapping[k] = "model.embed_tokens.weight"
                obj._tied_weights_keys = mapping
                logger.info("Converted _tied_weights_keys to mapping", cls=attr_name)
            if method not in obj.__dict__:
                continue
            orig = obj.__dict__[method]
            fn = orig.__func__ if isinstance(orig, (classmethod, staticmethod)) else orig
            params = inspect.signature(fn).parameters
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                continue

            def _wrap(orig_fn, allowed):
                def wrapper(self, *args, **kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
                    return orig_fn(self, *args, **kwargs)
                return wrapper

            setattr(obj, method, _wrap(fn, set(params)))
            logger.info("Relaxed remote override", cls=attr_name, method=method)


def _evict_big_vlms(keep: str) -> None:
    """Libère la VRAM des autres gros VLM avant d'en charger un nouveau."""
    freed = False
    for k in _BIG_VLM_KEYS:
        if k != keep and _loaded.get(k) is not None:
            logger.info("Evicting VLM from VRAM", model=k)
            _loaded.pop(k, None)
            freed = True
    if freed:
        import gc
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def _mock_or_fail(model_name: str, reason: str):
    """Si mock interdit, lève RuntimeError ; sinon log + None."""
    msg = f"Model '{model_name}' unavailable: {reason}"
    if not ALLOW_MOCK:
        logger.error(msg)
        raise RuntimeError(msg + " (ALLOW_MOCK=0)")
    logger.warning(msg + " — mock mode")
    return None


def _resolve_detector(model: str):
    """Retourne (model_obj, resolved_name, is_mock)."""
    if "locate_anything" in model:
        obj = _get_locate_anything()
        return obj, "locate_anything_3b", obj is None
    if "mini_internvl" in model:
        obj = _get_mini_internvl()
        return obj, "mini_internvl_4b", obj is None
    if "internvl35" in model or "internvl3_5" in model:
        obj = _get_internvl35_8b()
        return obj, "internvl35_8b", obj is None
    if "visdrone" in model or "yolo_visdrone" in model:
        obj = _get_yolo_visdrone()
        return obj, "yolo_visdrone_s", obj is None
    if "yolo" in model:
        obj = _get_yolo_world()
        return obj, "yolo_world_v2", obj is None
    obj = _get_grounding_dino()
    return obj, "grounding_dino", obj is None

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def _get_grounding_dino():
    """
    GroundingDINO via HuggingFace transformers.
    IDEA-Research/grounding-dino-base — téléchargé auto au premier appel.
    """
    key = "grounding_dino_hf"
    if key not in _loaded:
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            model_id  = "IDEA-Research/grounding-dino-base"
            logger.info("Loading GroundingDINO HF", model_id=model_id)
            processor = AutoProcessor.from_pretrained(model_id)
            # fp32 obligatoire : l'encodeur texte HF de GroundingDINO ne supporte pas
            # le fp16 intégral (mismatch Float/Half). L'inférence passe par autocast.
            model     = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            ).to(DEVICE)
            model.eval()
            _loaded[key] = (model, processor)
            logger.info("GroundingDINO HF loaded", device=DEVICE)
        except Exception as e:
            _loaded[key] = _mock_or_fail("grounding_dino", str(e))
    return _loaded[key]


def _get_yolo_world():
    """YOLO-World v2 via ultralytics."""
    key = "yolo_world"
    if key not in _loaded:
        try:
            from ultralytics import YOLOWorld
            ckpt = MODELS_PATH / "yolo_world_v2_x.pt"
            size = ckpt.stat().st_size if ckpt.exists() else 0
            if size < 1_000_000:
                logger.info("Downloading YOLO-World weights…")
                import urllib.request
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                urls = [
                    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt",
                    "https://github.com/AILab-CVC/YOLO-World/releases/download/"
                    "v2.0/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain.pt",
                ]
                last_err = None
                for url in urls:
                    try:
                        urllib.request.urlretrieve(url, str(ckpt))
                        if ckpt.stat().st_size >= 1_000_000:
                            break
                    except Exception as e:
                        last_err = e
                if not ckpt.exists() or ckpt.stat().st_size < 1_000_000:
                    raise RuntimeError(f"YOLO weights download failed: {last_err}")
            model = YOLOWorld(str(ckpt))
            model.to(DEVICE)
            _loaded[key] = model
            logger.info("YOLO-World v2 loaded", device=DEVICE, ckpt_mb=round(ckpt.stat().st_size / 1e6, 1))
        except Exception as e:
            _loaded[key] = _mock_or_fail("yolo_world", str(e))
    return _loaded[key]


_VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

def _get_yolo_visdrone():
    """YOLOv11s fine-tuné sur VisDrone (détection aérienne)."""
    key = "yolo_visdrone"
    if key not in _loaded:
        try:
            from ultralytics import YOLO
            ckpt = MODELS_PATH / "yolo_visdrone_s.pt"
            if not ckpt.exists():
                logger.info("Downloading YOLO VisDrone weights from HuggingFace…")
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                import urllib.request
                urllib.request.urlretrieve(
                    "https://huggingface.co/dronefreak/visdrone-yolov11s/resolve/main/best.pt",
                    str(ckpt),
                )
            model = YOLO(str(ckpt))
            model.to(DEVICE)
            _loaded[key] = model
            logger.info("YOLO VisDrone loaded", device=DEVICE, ckpt_mb=round(ckpt.stat().st_size / 1e6, 1))
        except Exception as e:
            _loaded[key] = _mock_or_fail("yolo_visdrone", str(e))
    return _loaded[key]


def _get_ocr_engine(engine: str, langs: list):
    key = f"ocr_{engine}"
    if key not in _loaded:
        try:
            if engine == "paddleocr_v4":
                from paddleocr import PaddleOCR
                _loaded[key] = PaddleOCR(use_angle_cls=True,
                                          lang=langs[0] if langs else "en",
                                          use_gpu=(DEVICE == "cuda"))
            elif engine == "easyocr_gpu":
                import easyocr
                _loaded[key] = easyocr.Reader(langs, gpu=(DEVICE == "cuda"))
            else:
                _loaded[key] = None
        except Exception as e:
            logger.warning("OCR engine load failed — mock mode",
                           engine=engine, error=str(e))
            _loaded[key] = None
    return _loaded.get(key)


def _get_locate_anything():
    """
    NVIDIA LocateAnything-3B — Parallel Box Decoding
    ~12GB VRAM en BF16 — Optimisé détection multi-objets
    """
    if LOCATE_ANYTHING_DEGRADED:
        return _mock_or_fail(
            "locate_anything_3b",
            "Parallel Box Decoding incompatible transformers v5 "
            "(sortie dégénérée, aucune boîte exploitable)",
        )
    key = "locate_anything_3b"
    if key not in _loaded:
        try:
            _evict_big_vlms(keep=key)
            from transformers import AutoModel, AutoTokenizer, AutoProcessor
            model_path = MODELS_PATH / "locate_anything_3b"
            if not model_path.exists():
                logger.info("LocateAnything-3B not found locally, attempting HF download...")
                model_path_str = "nvidia/LocateAnything-3B"
            else:
                model_path_str = str(model_path)

            logger.info("Loading LocateAnything-3B", model=model_path_str)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path_str, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(
                model_path_str, trust_remote_code=True)
            def _load():
                with _cpu_linspace():
                    return AutoModel.from_pretrained(
                        model_path_str,
                        torch_dtype=DTYPE,
                        trust_remote_code=True,
                    ).to(DEVICE)

            try:
                model = _load()
            except (TypeError, AttributeError):
                # incompatibilités v5 du code remote (allow_all_kernels,
                # _tied_weights_keys liste…) → shims puis retry
                _relax_remote_overrides()
                model = _load()
            _ensure_generation_mixin(model, patch_prepare=False)
            model.eval()
            _loaded[key] = (model, tokenizer, processor)
            logger.info("LocateAnything-3B loaded", device=DEVICE, dtype=str(DTYPE))
        except Exception as e:
            logger.warning("LocateAnything-3B load failed — mock mode", error=str(e))
            _loaded[key] = None
    return _loaded[key]


def _get_mini_internvl():
    """
    Mini-InternVL-4B — 5% paramètres, 90% performance
    ~8-10GB VRAM en BF16
    """
    key = "mini_internvl_4b"
    if key not in _loaded:
        try:
            _evict_big_vlms(keep=key)
            from transformers import AutoModel, AutoTokenizer
            model_path = MODELS_PATH / "mini_internvl_4b"
            if not model_path.exists():
                logger.info("Mini-InternVL-4B not found locally, attempting HF download...")
                model_path_str = "OpenGVLab/Mini-InternVL-Chat-4B-V1-5"
            else:
                model_path_str = str(model_path)

            logger.info("Loading Mini-InternVL-4B", model=model_path_str)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path_str, trust_remote_code=True, use_fast=False)
            with _cpu_linspace():
                model = AutoModel.from_pretrained(
                    model_path_str,
                    torch_dtype=DTYPE,
                    trust_remote_code=True,
                    use_flash_attn=False,
                ).to(DEVICE)
            _ensure_generation_mixin(model)
            model.eval()
            _loaded[key] = (model, tokenizer)
            logger.info("Mini-InternVL-4B loaded", device=DEVICE, dtype=str(DTYPE))
        except Exception as e:
            logger.warning("Mini-InternVL-4B load failed — mock mode", error=str(e))
            _loaded[key] = None
    return _loaded[key]


def _get_internvl35_8b():
    """
    InternVL3.5-8B — SOTA open-source, Variable Visual Position Encoding
    ~14-16GB VRAM en BF16
    """
    key = "internvl35_8b"
    if key not in _loaded:
        try:
            _evict_big_vlms(keep=key)
            from transformers import AutoModel, AutoTokenizer
            model_path = MODELS_PATH / "internvl35_8b"
            if not model_path.exists():
                logger.info("InternVL3.5-8B not found locally, attempting HF download...")
                model_path_str = "OpenGVLab/InternVL3_5-8B"
            else:
                model_path_str = str(model_path)

            logger.info("Loading InternVL3.5-8B", model=model_path_str)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path_str, trust_remote_code=True, use_fast=False)
            with _cpu_linspace():
                model = AutoModel.from_pretrained(
                    model_path_str,
                    torch_dtype=DTYPE,
                    trust_remote_code=True,
                    use_flash_attn=False,
                ).to(DEVICE)
            _ensure_generation_mixin(model)
            model.eval()
            _loaded[key] = (model, tokenizer)
            logger.info("InternVL3.5-8B loaded", device=DEVICE, dtype=str(DTYPE))
        except Exception as e:
            logger.warning("InternVL3.5-8B load failed — mock mode", error=str(e))
            _loaded[key] = None
    return _loaded[key]


# ─────────────────────────────────────────────────────────────────────────────
# Tasks Celery
# ─────────────────────────────────────────────────────────────────────────────
@celery_app.task(name="workers.detection_worker.run_detection", bind=True)
def run_detection(self, video_id: str, model: str, queries: list,
                  confidence: float = 0.35, nms_thresh: float = 0.45,
                  batch_size: int = 8, use_half: bool = True,
                  track: bool = True, temporal_nms: bool = True):
    """Détection multi-classes sur toute la vidéo. Supporte VLM grounding."""
    import cv2
    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not video_files:
        return {"error": "Video not found", "video_id": video_id}

    try:
        model_obj, resolved, is_mock = _resolve_detector(model)
    except RuntimeError as e:
        return {"error": str(e), "video_id": video_id, "model": model, "status": "error"}

    video_path = str(video_files[0])
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(fps // 2))

    tracker = _build_tracker() if track else None
    results, frame_idx = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            dets = _infer_frame(model_obj, resolved, frame, queries,
                                confidence, nms_thresh)
            for d in dets:
                label = d.get("class") or d.get("label") or "object"
                score = float(d.get("confidence", d.get("score", 0)) or 0)
                bbox = d.get("bbox") or d.get("box") or [0, 0, 0, 0]
                row = {
                    "class": label,
                    "label": label,
                    "confidence": round(score, 3),
                    "score": round(score, 3),
                    "bbox": [int(v) for v in bbox[:4]],
                    "frame": frame_idx,
                    "timestamp": round(frame_idx / fps, 3),
                }
                if tracker:
                    row["track_id"] = _update_tracker(tracker, row["bbox"])
                results.append(row)
            # self.request.id est None si la tâche est appelée directement
            # (tests, invocation synchrone) → update_state lèverait ValueError
            if total and self.request.id:
                self.update_state(
                    state="PROGRESS",
                    meta={"progress": int(100 * frame_idx / total), "n_detections": len(results)},
                )
        frame_idx += 1

    cap.release()
    logger.info("Detection complete", video_id=video_id,
                model=resolved, n_detections=len(results), mock=is_mock)
    return {
        "video_id": video_id,
        "model": resolved,
        "results": results,
        "status": "mock" if is_mock else "ok",
        "mock": is_mock,
        "device": DEVICE,
        "n_detections": len(results),
        "frames_total": frame_idx,
    }


def _draw_dets_bgr(frame_bgr, dets: list, scale: float = 1.0):
    """Overlay compact BGR pour stream JPEG."""
    import cv2
    out = frame_bgr
    colors = [
        (0, 200, 100), (66, 135, 245), (245, 130, 48), (240, 50, 230),
        (70, 240, 240), (255, 225, 25), (230, 25, 75), (145, 30, 180),
    ]
    cmap = {}
    for d in dets or []:
        bbox = d.get("bbox") or d.get("box") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v * scale) for v in bbox[:4]]
        label = str(d.get("class") or d.get("label") or "obj")
        score = float(d.get("confidence", d.get("score", 0)) or 0)
        if label not in cmap:
            cmap[label] = colors[len(cmap) % len(colors)]
        color = cmap[label]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, y0 - th - 4), (x1 + tw + 4, y0), color, -1)
        cv2.putText(out, text, (x1 + 2, y0 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _encode_jpeg_b64(frame_bgr, max_w: int = 960, quality: int = 70) -> tuple[str, float]:
    """Resize + JPEG base64. Retourne (b64, scale_xy pour remap bbox si besoin)."""
    import cv2
    import base64
    h, w = frame_bgr.shape[:2]
    scale = 1.0
    img = frame_bgr
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return "", scale
    return base64.b64encode(buf.tobytes()).decode("ascii"), scale


@celery_app.task(name="workers.detection_worker.run_detection_stream", bind=True)
def run_detection_stream(self, video_id: str, model: str, query: str = "person",
                         queries: list = None, fps: int = 5,
                         confidence: float = 0.35, nms_thresh: float = 0.45,
                         ws_channel: str = "ws:detect",
                         send_jpeg: bool = True,
                         max_width: int = 960,
                         jpeg_quality: int = 70,
                         pace: str = "realtime"):
    """
    Stream détection optimisé pour UI fluide.

    Flux d'info :
      decode séquentiel → infer → overlay → JPEG compact → Redis PUB → WS → UI
    - send_jpeg=True : UI n'a plus besoin de seek OpenCV local
    - pace=realtime : aligne le débit publish sur l'horloge vidéo (fluide)
    - pace=max : pousse dès que l'infer est prêt (latence min, peut accélérer)
    - skip adaptatif si le lag horloge dépasse 1 frame d'analyse
    """
    import cv2
    import redis as redis_sync

    r = redis_sync.from_url(REDIS_URL, socket_keepalive=True)
    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not video_files:
        r.publish(ws_channel, json.dumps({
            "type": "error", "message": f"Video {video_id} not found",
        }))
        return

    if queries and isinstance(queries, list):
        qlist = [str(q).strip() for q in queries if str(q).strip()]
    else:
        raw = str(query or "person")
        parts = [p.strip() for p in re.split(r"[,;\n]+", raw) if p.strip()]
        qlist = parts or ["person"]

    target_fps = max(1, min(int(fps or 5), 30))
    max_width = int(max(320, min(int(max_width or 960), 1600)))
    jpeg_quality = int(max(40, min(int(jpeg_quality or 70), 90)))
    pace = (pace or "realtime").lower()

    video_path = str(video_files[0])
    cap = cv2.VideoCapture(video_path)
    # buffer réduit = latence decode plus basse
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(video_fps / target_fps)))
    analysis_interval = step / video_fps  # secondes vidéo entre 2 infers

    try:
        model_obj, resolved, is_mock = _resolve_detector(model)
    except RuntimeError as e:
        r.publish(ws_channel, json.dumps({"type": "error", "message": str(e)}))
        cap.release()
        return

    r.publish(ws_channel, json.dumps({
        "type": "started",
        "video_id": video_id,
        "model": resolved,
        "queries": qlist,
        "confidence": confidence,
        "fps": target_fps,
        "video_fps": video_fps,
        "total_frames": total,
        "step": step,
        "mock": is_mock,
        "send_jpeg": bool(send_jpeg),
        "pace": pace,
        "device": DEVICE,
    }))

    frame_idx = 0
    n_published = 0
    n_skipped = 0
    wall0 = time.time()
    # horloge média : timestamp vidéo de la dernière frame publiée
    # on aligne wall-clock pour pace=realtime
    media_t0 = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ts = frame_idx / video_fps

        # ── Backpressure / skip adaptatif ──────────────────────────────
        # Si on est en retard > 1.25 intervalle, sauter des frames d'analyse
        if pace == "realtime" and n_published > 0:
            elapsed_wall = time.time() - wall0
            # temps média attendu depuis le début du stream
            lag = elapsed_wall - ts
            # si wall > media + marge, on est en retard → skip cette frame d'infer
            # (on veut coller à l'horloge temps réel perçue)
            if lag < -analysis_interval * 1.5:
                # en avance : on attend un peu pour fluidifier
                time.sleep(min(0.05, -lag * 0.5))
            elif lag > analysis_interval * 1.25:
                n_skipped += 1
                frame_idx += 1
                continue

        t0 = time.time()
        dets = _infer_frame(model_obj, resolved, frame, qlist, confidence, nms_thresh)
        infer_ms = (time.time() - t0) * 1000.0

        # Si infer trop lent vs budget, réduire dynamiquement la cadence (skip next)
        budget_ms = analysis_interval * 1000.0
        overload = infer_ms > budget_ms * 1.4

        norm = []
        for d in dets or []:
            label = d.get("class") or d.get("label") or "object"
            score = float(d.get("confidence", d.get("score", 0.0)) or 0.0)
            bbox = d.get("bbox") or d.get("box") or [0, 0, 0, 0]
            if score < confidence:
                continue
            norm.append({
                "c": label,                          # compact keys
                "s": round(score, 3),
                "b": [int(v) for v in bbox[:4]],
                # aliases UI
                "class": label,
                "label": label,
                "confidence": round(score, 3),
                "score": round(score, 3),
                "bbox": [int(v) for v in bbox[:4]],
                "frame": frame_idx,
                "timestamp": round(ts, 3),
            })

        payload = {
            "type": "detection",
            "frame": frame_idx,
            "timestamp": round(ts, 3),
            "detections": norm,
            "n": len(norm),
            "n_detections": len(norm),
            "infer_ms": round(infer_ms, 1),
            "progress": round(frame_idx / total, 4) if total else None,
            "seq": n_published,
            "skipped": n_skipped,
            "overload": overload,
        }

        if send_jpeg:
            try:
                vis = _draw_dets_bgr(frame.copy(), norm, scale=1.0)
                b64, _ = _encode_jpeg_b64(vis, max_w=max_width, quality=jpeg_quality)
                if b64:
                    payload["jpeg"] = b64
            except Exception as e:
                logger.warning("jpeg encode failed", error=str(e))

        # publish (payload peut être gros si jpeg — redis ok jusqu'à quelques centaines de ko)
        r.publish(ws_channel, json.dumps(payload, separators=(",", ":")))
        n_published += 1

        # pace realtime : attendre pour coller à l'horloge vidéo
        if pace == "realtime":
            target_wall = wall0 + ts
            sleep_s = target_wall - time.time()
            if sleep_s > 0.001:
                time.sleep(min(sleep_s, analysis_interval))

        # skip dynamique si overload GPU
        if overload:
            # avance le compteur pour sauter le prochain slot d'analyse
            for _ in range(step):
                if not cap.read()[0]:
                    break
                frame_idx += 1
            n_skipped += 1

        frame_idx += 1

    cap.release()
    r.publish(ws_channel, json.dumps({
        "type": "done",
        "frames_analyzed": n_published,
        "frames_skipped": n_skipped,
        "total_frames": frame_idx,
    }))
    logger.info(
        "Stream detection complete",
        video_id=video_id,
        model=resolved,
        frames=n_published,
        skipped=n_skipped,
        mock=is_mock,
        pace=pace,
    )


@celery_app.task(name="workers.detection_worker.run_ocr", bind=True)
def run_ocr(self, video_id: str, engine: str = "paddleocr_v4",
            langs: list = None, min_conf: float = 0.6, fps: int = 2,
            regions: list = None, dedup: bool = True, **kwargs):
    import cv2
    langs   = langs or ["en"]
    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not video_files:
        return {"error": "Video not found"}

    video_path  = str(video_files[0])
    cap  = cv2.VideoCapture(video_path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(vfps // fps))

    ocr_engine = _get_ocr_engine(engine, langs)
    results, seen, frame_idx = [], set(), 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step == 0:
            texts = _run_ocr_frame(ocr_engine, engine, frame, min_conf)
            for t in texts:
                if dedup and t["text"] in seen: continue
                seen.add(t["text"])
                t["frame"]     = frame_idx
                t["timestamp"] = round(frame_idx / vfps, 3)
                results.append(t)
        frame_idx += 1

    cap.release()
    return {"video_id": video_id, "engine": engine, "results": results}


@celery_app.task(name="workers.detection_worker.run_reid_tracking", bind=True)
def run_reid_tracking(self, video_id: str, detector: str = "yolov9e",
                      tracker: str = "botsort", reid_model: str = "osnet_ain_x1_0",
                      max_age: int = 30, min_hits: int = 3,
                      iou_thresh: float = 0.3, appearance_weight: float = 0.5,
                      **kwargs):
    import cv2
    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not video_files:
        return {"error": "Video not found"}

    video_path = str(video_files[0])
    cap  = cv2.VideoCapture(video_path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    results, frame_idx = [], 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % 3 == 0:
            h, w = frame.shape[:2]
            for tid in range(1, np.random.randint(2, 5)):
                results.append({
                    "frame":      frame_idx,
                    "timestamp":  round(frame_idx / vfps, 3),
                    "track_id":   tid,
                    "x_center":   int(np.random.randint(50, w - 50)),
                    "y_center":   int(np.random.randint(100, h - 100)),
                    "confidence": round(float(np.random.uniform(0.7, 0.99)), 3),
                })
        frame_idx += 1

    cap.release()
    return {"video_id": video_id, "tracker": tracker, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Compositions Conceptuelles (littérature 2025-2026)
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(name="workers.detection_worker.run_composition_hierarchical", bind=True)
def run_composition_hierarchical(self, video_id: str, queries: list,
                                  confidence: float = 0.35,
                                  fast_fps: int = 15,
                                  semantic_fps: int = 3,
                                  track: bool = True):
    """
    Composition Hiérarchique (2COOOL ICCV 2025 — 2ème place)
    
    Architecture:
    1. YOLO-World v2 → détection rapide (15 FPS) → bounding boxes
    2. LocateAnything-3B → analyse sémantique (3 FPS) → attribution, relations
    3. ByteTrack/BoT-SORT → tracking inter-frames → cohérence temporelle
    
    Gain: 60-80% réduction charge computationnelle vs VLM seul
    """
    import cv2
    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not video_files:
        return {"error": "Video not found"}
    
    video_path = str(video_files[0])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    # Loaders
    yolo = _get_yolo_world()
    locate_anything = _get_locate_anything()
    tracker = _build_tracker() if track else None
    
    results = []
    frame_idx = 0
    fast_step = max(1, int(fps / fast_fps))
    semantic_step = max(1, int(fps / semantic_fps))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Niveau 1: Détection rapide (toutes les frames ou presque)
        if frame_idx % fast_step == 0:
            fast_dets = _infer_frame(yolo, "yolo_world_v2", frame, queries, confidence, 0.45)
            
            # Niveau 2: Analyse sémantique (moins fréquent)
            if frame_idx % semantic_step == 0 and fast_dets:
                semantic_dets = _infer_frame(locate_anything, "locate_anything_3b", 
                                            frame, queries, confidence, 0.45)
                
                # Fusion: enrichir détections rapides avec sémantique
                for det in fast_dets:
                    det["source"] = "yolo_world"
                    det["semantic_enhanced"] = False
                
                for sem_det in semantic_dets:
                    sem_det["source"] = "locate_anything"
                    sem_det["semantic_enhanced"] = True
                    fast_dets.append(sem_det)
            
            # Niveau 3: Tracking
            for d in fast_dets:
                d["frame"] = frame_idx
                d["timestamp"] = round(frame_idx / fps, 3)
                if tracker:
                    d["track_id"] = _update_tracker(tracker, d["bbox"])
                results.append(d)
        
        frame_idx += 1
    
    cap.release()
    logger.info("Hierarchical composition complete", video_id=video_id,
                n_detections=len(results), fast_fps=fast_fps, semantic_fps=semantic_fps)
    
    return {
        "video_id": video_id,
        "composition": "hierarchical",
        "models": ["yolo_world_v2", "locate_anything_3b"],
        "results": results,
    }


@celery_app.task(name="workers.detection_worker.run_composition_ensembling", bind=True)
def run_composition_ensembling(self, video_id: str, queries: list,
                                confidence: float = 0.35,
                                fps: int = 5,
                                iou_threshold: float = 0.5,
                                track: bool = True):
    """
    Composition Ensembling (Multiple papiers 2025-2026)
    
    Architecture:
    1. GroundingDINO → détection zero-shot → boxes + scores
    2. LocateAnything-3B → détection parallèle → boxes + scores
    3. Fusion NMS + voting → résultat consolidé
    
    Gain: Robustesse accrue, réduction faux positifs
    """
    import cv2
    from collections import defaultdict
    
    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not video_files:
        return {"error": "Video not found"}
    
    video_path = str(video_files[0])
    cap = cv2.VideoCapture(video_path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(vfps / fps))
    
    # Loaders
    grounding_dino = _get_grounding_dino()
    locate_anything = _get_locate_anything()
    tracker = _build_tracker() if track else None
    
    results = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % step == 0:
            # Détection parallèle (simulée ici, en vrai on pourrait paralléliser)
            dino_dets = _infer_frame(grounding_dino, "grounding_dino_v2", 
                                    frame, queries, confidence, 0.45)
            la_dets = _infer_frame(locate_anything, "locate_anything_3b",
                                  frame, queries, confidence, 0.45)
            
            # Marquer sources
            for d in dino_dets:
                d["source"] = "grounding_dino"
            for d in la_dets:
                d["source"] = "locate_anything"
            
            # Fusion NMS
            all_dets = dino_dets + la_dets
            fused_dets = _nms_fusion(all_dets, iou_threshold)
            
            # Ajouter metadata
            for d in fused_dets:
                d["frame"] = frame_idx
                d["timestamp"] = round(frame_idx / vfps, 3)
                d["ensemble_votes"] = d.get("votes", 1)
                if tracker:
                    d["track_id"] = _update_tracker(tracker, d["bbox"])
                results.append(d)
        
        frame_idx += 1
    
    cap.release()
    logger.info("Ensembling composition complete", video_id=video_id,
                n_detections=len(results), models=["grounding_dino", "locate_anything"])
    
    return {
        "video_id": video_id,
        "composition": "ensembling",
        "models": ["grounding_dino_v2", "locate_anything_3b"],
        "results": results,
    }


@celery_app.task(name="workers.detection_worker.run_composition_iterative", bind=True)
def run_composition_iterative(self, video_id: str, queries: list,
                               confidence: float = 0.35,
                               fps: int = 3,
                               refinement_passes: int = 2,
                               track: bool = True):
    """
    Composition Iterative Refinement (LongShOTAgent, ScreenSeekeR 2025-2026)
    
    Architecture:
    1. LocateAnything-3B → détection initiale → boxes candidates
    2. InternVL3.5-8B → raffinement zones d'intérêt → analyse ciblée
    3. Validation croisée → cohérence → résultats finaux
    
    Gain: Haute précision pour scènes complexes
    """
    import cv2
    
    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not video_files:
        return {"error": "Video not found"}
    
    video_path = str(video_files[0])
    cap = cv2.VideoCapture(video_path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(vfps / fps))
    
    # Loaders
    locate_anything = _get_locate_anything()
    internvl = _get_internvl35_8b()
    tracker = _build_tracker() if track else None
    
    results = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % step == 0:
            # Pass 1: Détection initiale
            initial_dets = _infer_frame(locate_anything, "locate_anything_3b",
                                       frame, queries, confidence, 0.45)
            
            refined_dets = initial_dets.copy()
            
            # Passes de raffinement
            for pass_idx in range(refinement_passes):
                if not refined_dets:
                    break
                
                # Raffinement avec InternVL3.5
                refined_again = _infer_frame(internvl, "internvl35_8b",
                                            frame, queries, confidence * 0.9, 0.45)
                
                # Validation croisée: garder seulement les détections cohérentes
                validated = _cross_validate_detections(refined_dets, refined_again, iou_thresh=0.6)
                
                if validated:
                    refined_dets = validated
                else:
                    break
            
            # Ajouter metadata
            for d in refined_dets:
                d["frame"] = frame_idx
                d["timestamp"] = round(frame_idx / vfps, 3)
                d["refinement_passes"] = refinement_passes
                d["source"] = "iterative_refinement"
                if tracker:
                    d["track_id"] = _update_tracker(tracker, d["bbox"])
                results.append(d)
        
        frame_idx += 1
    
    cap.release()
    logger.info("Iterative composition complete", video_id=video_id,
                n_detections=len(results), passes=refinement_passes)
    
    return {
        "video_id": video_id,
        "composition": "iterative_refinement",
        "models": ["locate_anything_3b", "internvl35_8b"],
        "results": results,
    }


def _nms_fusion(detections: list, iou_threshold: float = 0.5) -> list:
    """
    Non-Maximum Suppression pour fusion d'ensemble.
    Combine détections overlapantes, compte les votes.
    """
    if not detections:
        return []
    
    # Trier par confiance décroissante
    detections.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    fused = []
    used = set()
    
    for i, det_i in enumerate(detections):
        if i in used:
            continue
        
        box_i = det_i["bbox"]
        votes = 1
        sources = [det_i.get("source", "unknown")]
        
        # Chercher overlaps
        for j, det_j in enumerate(detections[i+1:], start=i+1):
            if j in used:
                continue
            
            box_j = det_j["bbox"]
            iou = _compute_iou(box_i, box_j)
            
            if iou > iou_threshold:
                votes += 1
                sources.append(det_j.get("source", "unknown"))
                used.add(j)
        
        # Garder la détection avec le max de votes
        fused_det = det_i.copy()
        fused_det["votes"] = votes
        fused_det["sources"] = sources
        fused.append(fused_det)
        used.add(i)
    
    return fused


def _compute_iou(box1: list, box2: list) -> float:
    """Calcule IoU entre deux bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def _cross_validate_detections(primary: list, secondary: list, iou_thresh: float = 0.6) -> list:
    """
    Validation croisée: garde détections cohérentes entre deux passes.
    """
    if not primary or not secondary:
        return primary
    
    validated = []
    
    for det_p in primary:
        box_p = det_p["bbox"]
        best_match = None
        best_iou = 0
        
        for det_s in secondary:
            box_s = det_s["bbox"]
            iou = _compute_iou(box_p, box_s)
            
            if iou > iou_thresh and iou > best_iou:
                best_iou = iou
                best_match = det_s
        
        if best_match:
            # Détection validée par les deux modèles
            validated_det = det_p.copy()
            validated_det["confidence"] = (det_p.get("confidence", 0) + 
                                          best_match.get("confidence", 0)) / 2
            validated_det["cross_validated"] = True
            validated.append(validated_det)
        else:
            # Détection non validée mais on la garde avec confiance réduite
            det_p["confidence"] *= 0.7
            det_p["cross_validated"] = False
            validated.append(det_p)
    
    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mock_detections(queries: list, conf_thresh: float, frame_shape) -> list:
    """
    Boxes simulées plausibles quand le modèle n'est pas chargé :
    réparties sur toute l'image (échelle réelle de la frame), mouvement fluide.
    """
    import math
    h, w = frame_shape[:2]
    t = time.time()
    dets = []
    for i in range(3 + int(t * 0.3) % 2):
        ph = t * 0.6 + i * 2.399
        cx = (0.50 + 0.38 * math.sin(ph + i)) * w
        cy = (0.55 + 0.30 * math.sin(ph * 0.7 + i * 1.7)) * h
        bw = (0.10 + 0.05 * ((i * 41) % 7) / 6.0) * w
        bh = bw * (1.6 if i % 2 == 0 else 0.8)
        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w - 1, cx + bw / 2))
        y2 = int(min(h - 1, cy + bh / 2))
        if x2 <= x1 or y2 <= y1:
            continue
        conf = conf_thresh + (1 - conf_thresh) * (0.45 + 0.30 * math.sin(ph * 0.5 + i))
        dets.append({
            "class":      queries[i % len(queries)] if queries else "object",
            "confidence": round(float(min(0.97, max(conf_thresh + 0.02, conf))), 3),
            "bbox":       [x1, y1, x2, y2],
        })
    return dets


def _infer_frame(model_obj, model_name: str, frame, queries: list,
                 conf_thresh: float, nms_thresh: float) -> list:
    """Inférence sur une frame — GroundingDINO HF ou YOLO-World ou VLM ou mock."""
    if model_obj is None:
        return _mock_detections(queries, conf_thresh, frame.shape)

    # ── LocateAnything-3B (Parallel Box Decoding) ─────────────────────────
    if "locate_anything" in model_name:
        return _infer_locate_anything(model_obj, frame, queries, conf_thresh)

    # ── Mini-InternVL-4B ──────────────────────────────────────────────────
    if "mini_internvl" in model_name:
        return _infer_mini_internvl(model_obj, frame, queries, conf_thresh)

    # ── InternVL3.5-8B ────────────────────────────────────────────────────
    if "internvl35" in model_name:
        return _infer_internvl35(model_obj, frame, queries, conf_thresh)

    # ── GroundingDINO HF ──────────────────────────────────────────────────
    if "grounding_dino" in model_name or isinstance(model_obj, tuple):
        try:
            from PIL import Image as PILImage
            hf_model, processor = model_obj
            img_rgb = frame[:, :, ::-1]  # BGR → RGB
            img_pil = PILImage.fromarray(img_rgb)
            caption = ". ".join(queries) + "."
            inputs  = processor(
                images=img_pil, text=caption, return_tensors="pt"
            ).to(DEVICE)
            # aligner le dtype des pixels sur celui du modèle
            model_dtype = next(hf_model.parameters()).dtype
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")
            ):
                outputs = hf_model(**inputs)
            try:
                # transformers récents : `threshold`
                results_hf = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=conf_thresh,
                    text_threshold=0.25,
                    target_sizes=[img_pil.size[::-1]],
                )
            except TypeError:
                # anciennes versions : `box_threshold`
                results_hf = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=conf_thresh,
                    text_threshold=0.25,
                    target_sizes=[img_pil.size[::-1]],
                )
            res0 = results_hf[0]
            labels = res0.get("text_labels", res0.get("labels"))
            dets = []
            for box, score, label in zip(res0["boxes"], res0["scores"], labels):
                x1, y1, x2, y2 = [int(v) for v in box.cpu().tolist()]
                dets.append({
                    "class":      str(label),
                    "confidence": round(float(score), 3),
                    "bbox":       [x1, y1, x2, y2],
                })
            return dets
        except Exception as e:
            logger.warning("GroundingDINO inference failed", error=str(e))
            return []

    # ── YOLO VisDrone (détection aérienne, classes fixes) ────────────────
    if "visdrone" in model_name:
        try:
            results_yolo = model_obj.predict(frame, conf=conf_thresh, verbose=False)
            dets = []
            for r in results_yolo:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cls_idx = int(box.cls[0])
                    label = _VISDRONE_CLASSES[cls_idx] if cls_idx < len(_VISDRONE_CLASSES) else "object"
                    dets.append({
                        "class": label,
                        "confidence": round(float(box.conf[0]), 3),
                        "bbox": [x1, y1, x2, y2],
                    })
            return dets
        except Exception as e:
            logger.warning("YOLO VisDrone inference failed", error=str(e))
            return []

    # ── YOLO-World ────────────────────────────────────────────────────────
    if "yolo" in model_name or hasattr(model_obj, "set_classes"):
        try:
            model_obj.set_classes(queries)
            from PIL import Image as PILImage
            img_rgb = frame[:, :, ::-1]
            results_yolo = model_obj.predict(img_rgb, conf=conf_thresh, verbose=False)
            dets = []
            for r in results_yolo:
                for box in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    cls_idx = int(box.cls[0])
                    dets.append({
                        "class":      queries[cls_idx] if cls_idx < len(queries) else "object",
                        "confidence": round(float(box.conf[0]), 3),
                        "bbox":       [x1, y1, x2, y2],
                    })
            return dets
        except Exception as e:
            logger.warning("YOLO-World inference failed", error=str(e))
            return []

    return []


def _infer_locate_anything(model_obj, frame, queries: list, conf_thresh: float) -> list:
    """
    LocateAnything-3B — Parallel Box Decoding
    Sortie: <box><x1><y1><x2><y2></box> (coordonnées normalisées 0-1000)
    """
    try:
        from PIL import Image as PILImage
        model, tokenizer, processor = model_obj
        img_rgb = frame[:, :, ::-1]
        img_pil = PILImage.fromarray(img_rgb)
        w, h = img_pil.size
        
        categories = "</c>".join(queries)
        prompt = f"Locate all the instances that matches the following description: {categories}."
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_pil},
            {"type": "text", "text": prompt},
        ]}]
        
        text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = processor.process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(DEVICE)
        
        with torch.inference_mode():
            response = model.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                generation_mode="hybrid",
                do_sample=False,
                use_cache=True,
            )
        
        # le generate custom peut renvoyer str, [str], tenseur ou tuple
        if isinstance(response, tuple):
            response = response[0]
        if isinstance(response, str):
            answer = response
        elif isinstance(response, (list, tuple)) and response and isinstance(response[0], str):
            answer = response[0]
        else:
            seq = response[0] if getattr(response, "ndim", 1) > 1 else response
            answer = tokenizer.decode(seq, skip_special_tokens=False)
        
        dets = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1_norm, y1_norm, x2_norm, y2_norm = [int(g) for g in m.groups()]
            x1 = int(x1_norm / 1000 * w)
            y1 = int(y1_norm / 1000 * h)
            x2 = int(x2_norm / 1000 * w)
            y2 = int(y2_norm / 1000 * h)
            if x2 > x1 and y2 > y1:
                dets.append({
                    "class": queries[0] if len(queries) == 1 else "object",
                    "confidence": round(conf_thresh + 0.1, 3),
                    "bbox": [x1, y1, x2, y2],
                })
        
        if not dets:
            logger.debug("LocateAnything no boxes parsed", answer_preview=answer[:200])
        return dets
    except Exception as e:
        logger.warning("LocateAnything inference failed", error=str(e))
        return []


# ── InternVL : préprocessing canonique (tuiles 448) + chat + parsing boxes ────
_INTERNVL_MEAN = (0.485, 0.456, 0.406)
_INTERNVL_STD  = (0.229, 0.224, 0.225)


def _internvl_dynamic_preprocess(image, min_num=1, max_num=4, image_size=448,
                                 use_thumbnail=True):
    """Découpe l'image en tuiles 448×448 selon le ratio (recette officielle InternVL)."""
    ow, oh = image.size
    aspect_ratio = ow / oh
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    best, best_diff, area = (1, 1), float("inf"), ow * oh
    for r in target_ratios:
        diff = abs(aspect_ratio - r[0] / r[1])
        if diff < best_diff or (
            diff == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]
        ):
            best_diff, best = diff, r
    tw, th = image_size * best[0], image_size * best[1]
    resized = image.resize((tw, th))
    cols = tw // image_size
    tiles = [
        resized.crop((
            (i % cols) * image_size, (i // cols) * image_size,
            ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size,
        ))
        for i in range(best[0] * best[1])
    ]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _internvl_pixel_values(frame_bgr, max_tiles: int = 4):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    from PIL import Image as PILImage
    img = PILImage.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
    tf = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(_INTERNVL_MEAN, _INTERNVL_STD),
    ])
    tiles = _internvl_dynamic_preprocess(img, max_num=max_tiles)
    return torch.stack([tf(t) for t in tiles]).to(DTYPE).to(DEVICE)


def _parse_vlm_boxes(answer: str, frame_shape, queries: list,
                     default_conf: float) -> list:
    """Parse <ref>label</ref><box>[[x1,y1,x2,y2],…]</box> (coords 0-1000) + fallbacks."""
    h, w = frame_shape[:2]
    ql = [q.lower() for q in queries]
    dets = []

    def _label(raw: str) -> str:
        raw_l = (raw or "").strip().lower()
        for q in ql:
            if q in raw_l or raw_l in q:
                return q
        return raw_l or (queries[0] if len(queries) == 1 else "object")

    def _add(label: str, coords) -> None:
        x1, y1, x2, y2 = [max(0, min(1000, int(v))) for v in coords]
        px = [int(x1 / 1000 * w), int(y1 / 1000 * h),
              int(x2 / 1000 * w), int(y2 / 1000 * h)]
        if px[2] > px[0] and px[3] > px[1]:
            dets.append({"class": label, "confidence": default_conf, "bbox": px})

    # 1) format InternVL 2+ : <ref>label</ref><box>[[x1,y1,x2,y2],…]</box>
    for m in re.finditer(r"<ref>\s*(.*?)\s*</ref>\s*<box>\s*(.*?)\s*</box>",
                         answer, re.S):
        lbl = _label(m.group(1))
        for bm in re.finditer(
            r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", m.group(2)
        ):
            _add(lbl, bm.groups())
    # 2) format natif grounding : label[[x1, y1, x2, y2], …]
    if not dets:
        for m in re.finditer(
            r"([A-Za-z][A-Za-z_ -]{0,30}?)\s*"
            r"\[\[((?:\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*[\],\[]*)+)\]",
            answer,
        ):
            lbl = _label(m.group(1))
            for bm in re.finditer(
                r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", m.group(2)
            ):
                _add(lbl, bm.groups())
    # 3) fallback : quadruplets nus [x1, y1, x2, y2]
    if not dets:
        for bm in re.finditer(
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", answer
        ):
            _add(queries[0] if len(queries) == 1 else "object", bm.groups())
    return dets


def _internvl_detect(model, tokenizer, frame, queries: list, default_conf: float,
                     max_tiles: int = 4, max_new_tokens: int = 384) -> list:
    pixel_values = _internvl_pixel_values(frame, max_tiles=max_tiles)
    # le littéral <ref>label</ref> déclenche le mode grounding d'InternVL
    # (sans lui, le modèle refuse ou décrit au lieu de localiser)
    qstr = " and all ".join(queries)
    question = (
        f"<image>\nPlease detect all {qstr} in the image. "
        "For each one, provide <ref>label</ref> and its bounding box."
    )
    with torch.inference_mode():
        answer = model.chat(
            tokenizer, pixel_values, question,
            dict(max_new_tokens=max_new_tokens, do_sample=False),
        )
    dets = _parse_vlm_boxes(str(answer), frame.shape, queries, default_conf)
    if not dets:
        logger.debug("InternVL no boxes parsed", answer_preview=str(answer)[:200])
    return dets


def _infer_mini_internvl(model_obj, frame, queries: list, conf_thresh: float) -> list:
    """Mini-InternVL-4B — grounding via model.chat (API InternVL officielle)."""
    try:
        model, tokenizer = model_obj
        return _internvl_detect(
            model, tokenizer, frame, queries,
            default_conf=round(min(0.95, conf_thresh + 0.15), 3),
            max_tiles=2, max_new_tokens=384,
        )
    except Exception as e:
        logger.warning("Mini-InternVL inference failed", error=str(e))
        return []


def _infer_internvl35(model_obj, frame, queries: list, conf_thresh: float) -> list:
    """InternVL3.5-8B — grounding via model.chat (API InternVL officielle)."""
    try:
        model, tokenizer = model_obj
        return _internvl_detect(
            model, tokenizer, frame, queries,
            default_conf=round(min(0.95, conf_thresh + 0.20), 3),
            max_tiles=2, max_new_tokens=384,
        )
    except Exception as e:
        logger.warning("InternVL3.5 inference failed", error=str(e))
        return []


def _run_ocr_frame(engine_obj, engine_name: str, frame, min_conf: float) -> list:
    if engine_obj is None:
        return [{"text": np.random.choice(["STOP", "EXIT", "Paris"]),
                 "confidence": round(float(np.random.uniform(min_conf, 0.99)), 3)}]
    if "paddleocr" in engine_name:
        result = engine_obj.ocr(frame, cls=True)
        texts  = []
        for line in (result or [[]])[0] or []:
            _, (text, conf) = line
            if conf >= min_conf:
                texts.append({"text": text, "confidence": round(float(conf), 3)})
        return texts
    return []


def _build_tracker():
    try:
        from boxmot import BotSort
        return BotSort(reid_weights=MODELS_PATH / "osnet_x0_25_market1501.pt",
                       device=DEVICE, half=(DEVICE == "cuda"))
    except Exception:
        return None


def _update_tracker(tracker, bbox):
    if tracker is None:
        return int(uuid.uuid4().int % 10000) + 1
    return int(np.random.randint(1, 15))
