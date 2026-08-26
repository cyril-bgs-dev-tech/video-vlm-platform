"""
api/logging_config.py — Logging structuré JSON pour l'API et les workers Celery
Usage :
    from logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Video uploaded", video_id="abc123", size_mb=42.5)
"""
import logging
import sys
import os
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formateur JSON compact — compatible avec Loki, Datadog, CloudWatch."""

    LEVEL_MAP = {
        logging.DEBUG:    "debug",
        logging.INFO:     "info",
        logging.WARNING:  "warning",
        logging.ERROR:    "error",
        logging.CRITICAL: "critical",
    }

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   self.LEVEL_MAP.get(record.levelno, "info"),
            "logger":  record.name,
            "msg":     record.getMessage(),
            "module":  record.module,
            "line":    record.lineno,
        }

        # Ajouter les champs extra passés via logger.info("msg", extra={...})
        for k, v in record.__dict__.items():
            if k not in (
                "args", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno",
                "message", "module", "msecs", "msg", "name", "pathname",
                "process", "processName", "relativeCreated", "stack_info",
                "taskName", "thread", "threadName",
            ):
                payload[k] = v

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


class StructuredLogger(logging.LoggerAdapter):
    """Logger avec support des kwargs structurés."""

    # Attributs réservés par logging.LogRecord — ne jamais les écraser
    _RESERVED = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process",
        "taskName", "message", "asctime",
        # kwargs réservés de logger.log()
        "exc_info", "stack_info", "stacklevel", "extra",
    })

    def process(self, msg, kwargs):
        extra = dict(kwargs.pop("extra", {}) or {})
        # Absorber TOUS les kwargs custom (y compris noms réservés LogRecord)
        # en les renommant pour éviter TypeError: unexpected keyword argument
        rename = {
            "filename": "video_filename",
            "module": "mod_name",
            "name": "obj_name",
            "msg": "detail_msg",
            "args": "log_args",
            "pathname": "path_name",
            "lineno": "line_no",
            "thread": "thread_id",
            "process": "process_id",
            "message": "message_text",
            "asctime": "asctime_val",
            "created": "created_ts",
            "levelname": "level_name",
            "levelno": "level_no",
            "funcName": "func_name",
            "exc_text": "exc_text_val",
            "stack_info": "stack_info_val",
            "taskName": "task_name",
        }
        keep = {"exc_info", "stacklevel"}
        for k in list(kwargs.keys()):
            if k in keep:
                continue
            val = kwargs.pop(k)
            if k == "extra":
                continue
            safe_k = rename.get(k, k)
            if safe_k in self._RESERVED or safe_k in rename:
                safe_k = f"ctx_{k}"
            extra[safe_k] = val
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> StructuredLogger:
    """
    Retourne un logger structuré.

    Exemple d'utilisation :
        logger = get_logger(__name__)
        logger.info("Task started", task_id="abc", video_id="xyz")
        logger.error("GPU OOM", model="sam2", frame=342, exc_info=True)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        level   = os.getenv("LOG_LEVEL", "INFO").upper()

        use_json = os.getenv("LOG_FORMAT", "json").lower() == "json"
        if use_json:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            ))

        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.propagate = False

    return StructuredLogger(logger, extra={})


# ── Configuration globale au démarrage ───────────────────────────────────────
def configure_logging():
    """
    À appeler une seule fois au démarrage de l'application FastAPI ou du worker.
    Configure le niveau racine et les loggers tiers trop verbeux.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(message)s",  # Le formatter JSON s'en charge
    )

    # Réduire le bruit des bibliothèques tierces
    for noisy in ("uvicorn.access", "httpx", "httpcore", "PIL", "boto3",
                  "botocore", "s3transfer", "urllib3", "celery.utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Garder les logs Celery tasks et FastAPI visibles
    logging.getLogger("celery.task").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
