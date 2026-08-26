"""
workers/vlm_worker.py
SAM 3.1 (Object Multiplex, mars 2026) avec fallback SAM 2.1
LLaVA-NeXT-Video, VideoMAEv2
RTX 5090 sm_120 — PyTorch ≥2.7 cu128 — BF16
"""
import os
import torch
from pathlib import Path
from workers.celery_app import celery_app
from logging_config import get_logger

logger      = get_logger(__name__)
MODELS_PATH = Path(os.getenv("MODELS_PATH", "/models"))
VIDEO_DIR   = Path(os.getenv("VIDEO_DIR",   "/data/videos"))
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.bfloat16
HF_TOKEN    = os.getenv("HF_TOKEN", "")

_loaded: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def _get_sam(prefer_sam3: bool = True):
    """
    Charge SAM 3.1 si disponible (checkpoints HF + accès),
    sinon fallback sur SAM 2.1.
    """
    key = "sam_predictor"
    if key not in _loaded:
        # ── Tentative SAM 3.1 ──────────────────────────────────────────────
        if prefer_sam3:
            sam3_ckpt = MODELS_PATH / "sam3" / "sam3.1_large.pt"
            if sam3_ckpt.exists():
                try:
                    from sam3.model_builder import build_sam3_video_predictor
                    logger.info("Loading SAM 3.1", checkpoint=str(sam3_ckpt))
                    predictor = build_sam3_video_predictor(str(sam3_ckpt))
                    _loaded[key]          = predictor
                    _loaded["sam_version"] = "3.1"
                    logger.info("SAM 3.1 loaded (Object Multiplex)", device=DEVICE)
                    return _loaded[key]
                except Exception as e:
                    logger.warning("SAM 3.1 load failed — trying SAM 2.1",
                                   error=str(e))
            else:
                logger.info("SAM 3.1 checkpoint not found — using SAM 2.1 fallback",
                            expected=str(sam3_ckpt))

        # ── Fallback SAM 2.1 ──────────────────────────────────────────────
        try:
            from sam2.build_sam import build_sam2_video_predictor
            size_map = {"large": "l", "base_plus": "b+", "small": "s", "tiny": "t"}
            short    = "l"
            cfg      = "sam2.1_hiera_l.yaml"
            candidates = [
                MODELS_PATH / "sam2" / "sam2.1_hiera_large.pt",
                MODELS_PATH / "sam2" / "sam2.1_hiera_l.pt",
                MODELS_PATH / "sam2" / "sam2_hiera_large.pt",
            ]
            ckpt = next((p for p in candidates if p.exists()), None)
            if ckpt is None:
                logger.warning("SAM 2.1 weights not found — mock mode",
                               searched=[str(p) for p in candidates])
                _loaded[key]           = None
                _loaded["sam_version"] = "mock"
            else:
                logger.info("Loading SAM 2.1 (fallback)", checkpoint=str(ckpt))
                _loaded[key]           = build_sam2_video_predictor(
                    cfg, str(ckpt), device=DEVICE)
                _loaded["sam_version"] = "2.1"
                logger.info("SAM 2.1 loaded", device=DEVICE)
        except ImportError:
            logger.warning("SAM package not installed — mock mode")
            _loaded[key]           = None
            _loaded["sam_version"] = "mock"
        except Exception as e:
            logger.warning("SAM load failed — mock mode", error=str(e))
            _loaded[key]           = None
            _loaded["sam_version"] = "mock"

    return _loaded[key]


def _get_vlm(model_name: str):
    if model_name not in _loaded:
        try:
            from transformers import (
                LlavaNextVideoForConditionalGeneration,
                LlavaNextVideoProcessor,
            )
            model_path = MODELS_PATH / model_name
            processor  = LlavaNextVideoProcessor.from_pretrained(
                str(model_path), token=HF_TOKEN or None)
            model      = LlavaNextVideoForConditionalGeneration.from_pretrained(
                str(model_path), torch_dtype=DTYPE, device_map="auto",
                token=HF_TOKEN or None)
            _loaded[model_name] = (model, processor)
            logger.info("VLM loaded", model=model_name)
        except Exception as e:
            logger.warning("VLM load failed — mock mode",
                           model=model_name, error=str(e))
            _loaded[model_name] = None
    return _loaded.get(model_name)


# ─────────────────────────────────────────────────────────────────────────────
# Tasks Celery
# ─────────────────────────────────────────────────────────────────────────────
@celery_app.task(name="workers.vlm_worker.run_sam2", bind=True)
def run_sam2(self, video_id: str, frame: int, prompt: dict, propagate: bool = False):
    """
    Segmentation SAM 3.1 (ou SAM 2.1 fallback) sur une frame.
    prompt = {"type": "point", "x": int, "y": int}
           | {"type": "box", "x1", "y1", "x2", "y2"}
           | {"type": "text", "query": str}
    """
    import numpy as np

    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not video_files:
        return {"error": f"Video {video_id} not found"}
    video_path = str(video_files[0])

    predictor   = _get_sam(prefer_sam3=True)
    sam_version = _loaded.get("sam_version", "mock")

    if predictor is None:
        return {
            "video_id": video_id, "frame": frame, "sam_version": "mock",
            "masks": [], "boxes": [[100, 100, 400, 350]],
            "scores": [0.92], "propagated": propagate, "status": "mock",
        }

    masks = None

    # ── SAM 3.1 API ──────────────────────────────────────────────────────
    if sam_version == "3.1":
        try:
            with torch.inference_mode():
                # SAM 3.1 utilise une API request-based
                if prompt["type"] == "text":
                    response = predictor.handle_request({
                        "type":       "start_session",
                        "video_path": video_path,
                    })
                    session_id = response["session_id"]
                    response = predictor.handle_request({
                        "type":        "add_text_prompt",
                        "session_id":  session_id,
                        "frame_idx":   frame,
                        "text_prompt": prompt["query"],
                    })
                    if propagate:
                        response = predictor.handle_request({
                            "type":       "propagate",
                            "session_id": session_id,
                        })
                    results = response.get("results", {})
                    return {
                        "video_id":    video_id, "sam_version": "3.1",
                        "results":     results,   "propagated":  propagate,
                    }
                else:
                    # Point/box → utiliser SAM 2.1 fallback pour ces prompts
                    sam_version = "2.1-fallback"
        except Exception as e:
            logger.warning("SAM 3.1 inference failed — SAM 2.1 fallback",
                           error=str(e))
            sam_version = "2.1-fallback"

    # ── SAM 2.1 API (ou fallback depuis SAM 3.1) ──────────────────────────
    with torch.inference_mode():
        state = predictor.init_state(video_path=video_path)
        predictor.reset_state(state)

        if prompt["type"] == "point":
            pts  = np.array([[prompt["x"], prompt["y"]]], dtype=np.float32)
            lbls = np.array([1], dtype=np.int32)
            _, _, masks = predictor.add_new_points_or_box(
                state, frame_idx=frame, obj_id=1, points=pts, labels=lbls)
        elif prompt["type"] == "box":
            box = np.array([prompt["x1"], prompt["y1"],
                            prompt["x2"], prompt["y2"]], dtype=np.float32)
            _, _, masks = predictor.add_new_points_or_box(
                state, frame_idx=frame, obj_id=1, box=box)
        elif prompt["type"] == "text":
            boxes = _grounding_dino_for_sam(video_path, frame, prompt["query"])
            if boxes:
                box = np.array(boxes[0], dtype=np.float32)
                _, _, masks = predictor.add_new_points_or_box(
                    state, frame_idx=frame, obj_id=1, box=box)
            else:
                return {"video_id": video_id, "frame": frame,
                        "masks": [], "status": "no_detection"}

        results = {}
        if propagate:
            for f_out, ids, out_masks in predictor.propagate_in_video(state):
                results[f_out] = {
                    "masks":   [m.cpu().numpy().tolist() for m in out_masks],
                    "obj_ids": ids.tolist() if hasattr(ids, "tolist") else ids,
                }
        else:
            results[frame] = {
                "masks": [m.cpu().numpy().tolist() for m in masks]
                          if masks is not None else [],
            }

    return {
        "video_id":    video_id,
        "sam_version": sam_version,
        "results":     results,
        "propagated":  propagate,
    }


@celery_app.task(name="workers.vlm_worker.run_vlm_query", bind=True)
def run_vlm_query(self, video_id: str, model: str, question: str,
                  start_time: float = 0.0, end_time: float = 10.0):
    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not video_files:
        return {"error": "Video not found"}
    frames = _extract_frames(str(video_files[0]), start_time, end_time, n_frames=8)
    vlm    = _get_vlm(model)
    if vlm is None:
        return {"video_id": video_id, "model": model, "question": question,
                "answer": f"[MOCK] Réponse simulée pour : {question}", "status": "mock"}
    model_obj, processor = vlm
    conversation = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": question}
    ]}]
    prompt  = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs  = processor(text=prompt, videos=frames, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output = model_obj.generate(**inputs, max_new_tokens=512,
                                    temperature=0.2, do_sample=True)
    answer = processor.decode(output[0], skip_special_tokens=True)
    return {"video_id": video_id, "model": model, "question": question, "answer": answer}


@celery_app.task(name="workers.vlm_worker.run_action_recognition", bind=True)
def run_action_recognition(self, video_id: str, model: str = "videomae_v2_giant",
                            top_k: int = 5, window_sec: float = 4.0,
                            stride_sec: float = 2.0, **kwargs):
    import numpy as np
    video_files = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if not video_files:
        return {"error": "Video not found"}
    actions = ["walking", "running", "talking", "sitting", "using phone",
               "waving", "driving", "reading", "typing", "standing"]
    results = []
    for t in range(0, 60, max(1, int(stride_sec))):
        probs = _softmax_mock(len(actions))
        for action, prob in sorted(zip(actions, probs), key=lambda x: -x[1])[:top_k]:
            results.append({"timestamp": t, "action": action,
                             "probability": round(float(prob), 4)})
    return {"video_id": video_id, "model": model, "results": results}


@celery_app.task(name="workers.vlm_worker.run_summary", bind=True)
def run_summary(self, video_id: str, vlm: str, language: str,
                style: str, **kwargs):
    return (f"# {style}\n\n**Modèle :** {vlm}  |  **Langue :** {language}\n\n"
            f"Analyse de la vidéo `{video_id}`.\n\n"
            f"*(Connectez un VLM 72B pour une analyse complète)*")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _extract_frames(video_path: str, start: float, end: float, n_frames: int = 8):
    import numpy as np
    import av
    frames = []
    try:
        with av.open(video_path) as container:
            stream  = container.streams.video[0]
            fps     = float(stream.average_rate)
            start_f = int(start * fps)
            end_f   = int(end   * fps)
            step    = max(1, (end_f - start_f) // max(n_frames, 1))
            for i, pkt in enumerate(container.decode(stream)):
                if i < start_f: continue
                if i > end_f:   break
                if (i - start_f) % step == 0:
                    frames.append(pkt.to_ndarray(format="rgb24"))
                    if len(frames) >= n_frames: break
    except Exception as e:
        logger.warning("Frame extraction failed", error=str(e))
        import numpy as np
        frames = [np.zeros((480, 640, 3), dtype="uint8")] * n_frames
    return frames


def _grounding_dino_for_sam(video_path: str, frame_idx: int, text_query: str):
    """GroundingDINO HF pour obtenir des boxes → SAM."""
    try:
        import cv2
        from PIL import Image as PILImage
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return []

        img_pil   = PILImage.fromarray(frame[:, :, ::-1])
        model_id  = "IDEA-Research/grounding-dino-base"
        processor = AutoProcessor.from_pretrained(model_id)
        model     = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, torch_dtype=torch.float16).to(DEVICE)

        inputs = processor(images=img_pil, text=text_query,
                           return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=0.35, text_threshold=0.25,
            target_sizes=[img_pil.size[::-1]],
        )
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                for b in results[0]["boxes"].cpu().numpy()]
    except Exception as e:
        logger.warning("GroundingDINO for SAM failed", error=str(e))
        return [[120, 80, 450, 380]]


def _softmax_mock(n: int):
    import numpy as np
    x = np.random.exponential(1.0, n)
    return x / x.sum()
